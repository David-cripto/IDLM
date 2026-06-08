import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils
import hydra.utils
import models
from ipdb import set_trace as debug


def _maybe_save_periodic_checkpoint(module):
  # Periodic checkpointing is handled by main.PeriodicStepCheckpoint.
  return


def _conditional_valid_tokens(config, valid_tokens):
  if getattr(config.data, 'train', None) == 'tiny_gsm':
    return valid_tokens
  return None


def _apply_top_p_mask(log_probs: torch.Tensor, top_p: float, neg_infinity: float):
    """
    log_probs: (B, V) log-probabilities (already log_softmax'd)
    Returns: log_probs with tokens outside nucleus set to -inf.
    """
    if top_p is None or top_p >= 1.0:
        return log_probs  # p=1 -> no change -> same results as before

    # Convert to probs for sorting / cumulative mass
    probs = log_probs.exp()  # (B, V)

    # Sort probs descending
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # (B, V)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, V)

    # Keep smallest set with cumulative mass >= top_p
    # We mark tokens to REMOVE where cum_probs > top_p, but ensure at least 1 token kept.
    sorted_remove = cum_probs > top_p
    sorted_remove[..., 0] = False  # always keep the top token

    # Map remove-mask back to original vocab positions
    remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_idx, sorted_remove)  # (B, V)

    # Mask out removed tokens in log space
    log_probs = log_probs.masked_fill(remove, neg_infinity)
    return log_probs

class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step, train_mode=None,
          valid_tokens=None):
    del current_accumulation_step, train_mode, valid_tokens
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    for i in range(num_pred_tokens):
      output = self.backbone(x[:, :i + 1], None)
      output[:, :, self.mask_index] = self.neg_infinity
      # log-probs for the next token
      log_probs = output[:, -1, :].log_softmax(-1)  # (B, V)

      log_probs = _apply_top_p_mask(log_probs, top_p=0.95, neg_infinity=self.neg_infinity)

      y = (log_probs + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()
    self.is_adversarial_distill = self.config.adversarial_distill.is_distill
    if self.is_adversarial_distill:
      self.automatic_optimization = False
      self.teacher = copy.deepcopy(self.backbone)
      utils.freeze_model(self.teacher)
      
      self.fake_model = copy.deepcopy(self.backbone)
      utils.activate_model(self.fake_model)
      utils.activate_model(self.backbone)
      if self.ema is not None:
        del self.ema
        self.ema = models.ema.ExponentialMovingAverage(
          self.backbone.parameters(),
          decay=self.config.training.ema)

        self.ema_fake = models.ema.ExponentialMovingAverage(
          self.fake_model.parameters(),
          decay=self.config.training.ema)
      self.training_step_counter = 0
      self.accum_batches = self.config.adversarial_distill.accum_batches 
      self._fake_accum_counter = 0
      self._student_accum_counter = 0
      self.fake_frequency = self.config.adversarial_distill.fake_frequency
      self.fake_iter = 0 
      self.is_generator_step = False
      self.total_loss_student = 0
      self.total_loss_fake = 0
      self.f_loss_ar = []
  
  def configure_optimizers(self):
    if self.config.adversarial_distill.is_distill:
      optimizer_student = torch.optim.AdamW(
        self.backbone.parameters(),
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
              self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)
      
      optimizer_fake = torch.optim.AdamW(
        self.fake_model.parameters(),
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
              self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)

      scheduler_student = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer_student)
      scheduler_student_dict = {'scheduler': scheduler_student, 'name': 'trainer/lr'}

      scheduler_fake = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer_fake)
      scheduler_fake_dict = {'scheduler': scheduler_fake, 'name': 'trainer/lr_fake'}
      return [optimizer_student, optimizer_fake], [scheduler_student_dict, scheduler_fake_dict]
    else:
      return super().configure_optimizers()

  def _validate_configuration(self):
    # ancestral sampling isn't desirable because it's slow
    assert self.sampler == 'ancestral_cache'

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def on_train_start(self):
    if self.is_adversarial_distill:
      self.ema_fake.move_shadow_params_to_device(self.device)
      utils.activate_model(self.fake_model)
      utils.freeze_model(self.backbone)
    super().on_train_start()

  def on_load_checkpoint(self, checkpoint):
    if self.is_adversarial_distill:
      state_dict = collections.OrderedDict(checkpoint['state_dict'])
      for key, value in list(state_dict.items()):
        if key.startswith('backbone.'):
          suffix = key[len('backbone.'):]
          state_dict.setdefault(f'teacher.{suffix}', value)
          state_dict.setdefault(f'fake_model.{suffix}', value)
      checkpoint['state_dict'] = state_dict
    super().on_load_checkpoint(checkpoint)
    if self.is_adversarial_distill and self.ema_fake is not None:
      ema_fake_state = checkpoint.get('ema_fake', checkpoint.get('ema'))
      if ema_fake_state is not None:
        self.ema_fake.load_state_dict(ema_fake_state)

  def on_save_checkpoint(self, checkpoint):
    super().on_save_checkpoint(checkpoint)
    if self.is_adversarial_distill and self.ema_fake is not None:
      checkpoint['ema_fake'] = self.ema_fake.state_dict()

  def _preserve_prompt_probs(self, probs, x0, valid_tokens):
    if valid_tokens is None:
      return probs
    clean_probs = F.one_hot(x0, num_classes=probs.shape[-1]).to(probs.dtype)
    return torch.where(valid_tokens[..., None].bool(), probs, clean_probs)

  def _preserve_prompt_tokens(self, tokens, x0, valid_tokens):
    if valid_tokens is None:
      return tokens
    return torch.where(valid_tokens.bool(), tokens, x0)

  def multistep_generation(self, x0, valid_tokens, current_accumulation_step):
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)

    log_x_theta = self.forward(xt, sigma=sigma)
    student_probs = log_x_theta.exp()
    return self._preserve_prompt_probs(student_probs, x0, valid_tokens)

  def q_xt_with_probs(self, student_probs, alpha_t, x0=None, valid_tokens=None):
    mask_probs = torch.zeros_like(student_probs)
    mask_probs[:, :, self.mask_index] = 1
    noise_probs = alpha_t[:, :, None] * student_probs + (1 - alpha_t[:, :, None])*(mask_probs)
    if x0 is not None and valid_tokens is not None:
      noise_probs = self._preserve_prompt_probs(noise_probs, x0, valid_tokens)
    return noise_probs

  def nll_per_token_with_probs(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, valid_tokens=None):
    del xt
    loss = (x0 * log_x_theta).sum(-1)
    loss = loss * dalpha_t / (1 - alpha_t)
    if valid_tokens is not None:
      loss = loss * valid_tokens
      denom = valid_tokens.sum(-1).clamp_min(1)
      return loss.sum(-1) / denom
    return loss.sum(-1)

  def generator_loss(self, x0, valid_tokens, current_accumulation_step):
    student_probs = self.multistep_generation(x0, valid_tokens, current_accumulation_step)
    
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)
    sigma = self._process_sigma(sigma)

    probs_xt = self.q_xt_with_probs(student_probs, alpha_t, x0=x0, valid_tokens=valid_tokens)
    xt = trainer_base.sample_categorical(probs_xt)
    xt = self._preserve_prompt_tokens(xt, x0, valid_tokens)
    
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_teacher = self.teacher(xt, sigma)
    logits_teacher = self._process_model_output(model_output=model_output_teacher, xt=xt, sigma=sigma)
    teacher_loss = self.nll_per_token_with_probs(logits_teacher, xt, student_probs, 
                                alpha_t=alpha_t, dalpha_t=dalpha_t,
                                valid_tokens=valid_tokens)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_fake = self.fake_model(xt, sigma)
    logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
    fake_loss = self.nll_per_token_with_probs(logits_fake, xt, student_probs, 
                                alpha_t=alpha_t, dalpha_t=dalpha_t,
                                valid_tokens=valid_tokens)
    return teacher_loss - fake_loss

  def fake_loss(self, x0, valid_tokens, current_accumulation_step):
    with torch.no_grad():
      student_probs = self.multistep_generation(x0, valid_tokens, current_accumulation_step)

    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)
    sigma = self._process_sigma(sigma)
    probs_xt = self.q_xt_with_probs(student_probs, alpha_t, x0=x0, valid_tokens=valid_tokens)
    xt = trainer_base.sample_categorical(probs_xt)
    xt = self._preserve_prompt_tokens(xt, x0, valid_tokens)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_fake = self.fake_model(xt, sigma)
    logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
    fake_loss = self.nll_per_token_with_probs(logits_fake, xt, student_probs, 
                                alpha_t=alpha_t, dalpha_t=dalpha_t,
                                valid_tokens=valid_tokens)

    return fake_loss

  def training_step(self, batch, batch_idx):
    if self.config.adversarial_distill.is_distill:
      optimizer_student, optimizer_fake = self.optimizers()
      scheduler_student, scheduler_fake = self.lr_schedulers()
      current_accumulation_step = (batch_idx % self.accum_batches)
      finish_iter = False

      (input_tokens, output_tokens, valid_tokens) = self._process_model_input(batch['input_ids'], 
                                                                              batch['attention_mask'])
      valid_tokens = _conditional_valid_tokens(self.config, valid_tokens)
      if self.is_generator_step:
        self.toggle_optimizer(optimizer_student)
        g_loss = self.generator_loss(input_tokens, valid_tokens, current_accumulation_step)
        g_loss = g_loss.mean()
        g_loss = g_loss / self.accum_batches
        self.manual_backward(g_loss)

        self.untoggle_optimizer(optimizer_student)

        self._student_accum_counter += 1
        self.total_loss_student += g_loss.detach()
        if self._student_accum_counter == self.accum_batches:
          self._student_accum_counter = 0 
          optimizer_student.step()
          self.ema.update(self.backbone.parameters())
          optimizer_student.zero_grad()
          finish_iter = True
      else:
        self.toggle_optimizer(optimizer_fake)
        f_loss = self.fake_loss(input_tokens, valid_tokens, current_accumulation_step)
        f_loss = f_loss.mean()
        f_loss = f_loss / self.accum_batches
        self.manual_backward(f_loss)

        self.untoggle_optimizer(optimizer_fake)
        self._fake_accum_counter += 1
        self.total_loss_fake += f_loss.detach()
        if self._fake_accum_counter == self.accum_batches:
          self._fake_accum_counter = 0 
          optimizer_fake.step()
          self.ema_fake.update(self.fake_model.parameters())
          optimizer_fake.zero_grad()
          finish_iter = True

      if finish_iter:
        _maybe_save_periodic_checkpoint(self)

      if finish_iter and self.is_generator_step:
        self.is_generator_step = False
        utils.activate_model(self.fake_model)
        utils.freeze_model(self.backbone)
        optimizer_fake.zero_grad()
        self.log_dict(
            {"student_loss": self.total_loss_student, "fake_loss": torch.stack(self.f_loss_ar).mean()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        scheduler_student.step()
        scheduler_fake.step()
        self.total_loss_student = 0
        self.f_loss_ar.clear()
      elif finish_iter:
        self.fake_iter += 1
        self.f_loss_ar.append(self.total_loss_fake)
        self.total_loss_fake = 0
        if self.fake_iter == self.fake_frequency:
          self.is_generator_step = True
          self.fake_iter = 0
          utils.activate_model(self.backbone)
          utils.freeze_model(self.fake_model)
          optimizer_student.zero_grad()

    else:
      return super().training_step(batch, batch_idx)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0

    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x 
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(
      model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_index).to(
      model_output.dtype)[:, :, None]
    model_output = (
      masked_score * masked_indices
      + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class D3PMAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.noise.type == 'log-linear'
    assert self.parameterization == 'mean'

  def _process_model_output(self, model_output, xt, sigma):
    del xt 
    del sigma
    if self.subs_masking:
      model_output[:, :, self.mask_index] += self.neg_infinity
    return model_output.log_softmax(dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del dalpha_t
    assert not low_var
    dt = 1 / self.T
    t = 1 - alpha_t  # Only valid for log-linear schedule.
    t = t.clamp(0., 1.0 - 1e-4)
    alpha_t = alpha_t + torch.zeros_like(xt)
    alpha_s = t - dt + torch.zeros_like(xt)
    assert alpha_s.shape == xt.shape
    assert alpha_t.shape == xt.shape
    log_x_theta_at_x0 = torch.gather(
      log_x_theta, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(
      alpha_s * x_theta_at_m / (t - dt) + 1)
    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
    return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.predictor == 'analytic'

  def _get_score(self, x, sigma):
    return self.forward(x, sigma).exp()

  def _process_model_output(self, model_output, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(model_output.dtype)
    # logits shape
    # (batch_size, context_length, vocab_size)
    model_output = (model_output
                    - esigm1_log[:, None, None]
                    - np.log(model_output.shape[-1] - 1))
    # The below scatter operation sets the log score
    # for the input word to 0.
    model_output = torch.scatter(
      model_output, -1, xt[..., None],
      torch.zeros_like(model_output[..., :1]))
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    """Computes the SEDD loss for the Absorbing State Diffusion.

    Args:
      log_x_theta: float torch.Tensor with shape (batch_size,
          context_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          context_length), input.
      x0: int torch.Tensor with shape (batch_size,
          context_length), input.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
          time derivative of signal level.
      low_var: bool, low variance loss during training.
    
    Returns:
      loss with shape (batch_size, context_length).
    """
    assert not low_var
    masked_indices = xt == self.mask_index
    sigma = self._sigma_from_alphat(alpha_t)
    dsigma = - dalpha_t / alpha_t

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_x_theta[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_x_theta[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return dsigma * entropy


class DUO_BASE(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    return model_output.log_softmax(dim=-1)

  def _compute_posterior(self, x, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot),
        or model's predicted `x_theta` of shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    if self.config.sampling.use_float64:
      x = x.to(torch.float64)
    if alpha_s.ndim == 2:
      alpha_s = alpha_s.unsqueeze(-1)
    if alpha_t.ndim == 2:
      alpha_t = alpha_t.unsqueeze(-1)
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size).to(
      self.dtype).to(self.device)
    return (
      (alpha_t * self.vocab_size * x * xt_one_hot + (
        alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
          1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
            alpha_t * self.vocab_size * torch.gather(
              x, -1, xt[..., None]) + (1 - alpha_t)))

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    term1 = self.vocab_size * (1 / xbar_xt
                                - 1 / xbar_theta_xt)
    
    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False):
    del p_x0
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    sigma_t = self._sigma_from_alphat(alpha_t)
    assert alpha_t.ndim == 2
    
    q_xs = self._compute_posterior(
      x=self.forward(x, sigma_t).exp(),
      xt=x,
      alpha_s=alpha_s,
      alpha_t=alpha_t)
    if self.p_nucleus < 1:
      q_xs = utils.top_k_top_p_filtering(
        q_xs.log(), top_p=self.p_nucleus)
    return None, trainer_base.sample_categorical(q_xs)


class Integral(torch.autograd.Function):
  """
  torch module calculating UDLM's p_t 
  """

  @staticmethod
  def forward(ctx, gamma_t, data):
    gamma_max = data['gamma_max']
    gamma_min = data['gamma_min']
    if (gamma_t.max() > gamma_max) or (
      gamma_t.min() < gamma_min):
      print('max:{} {}'.format(gamma_t.max(), gamma_max))
      print('min:{} {}'.format(gamma_t.min(), gamma_min))
      gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
    indices = torch.round(
      (data['num_points'] - 1) * (gamma_t - gamma_min) / (
          gamma_max - gamma_min)).long()
    grad_pt = data['grad_pt']
    ctx.grad_pt = grad_pt[indices]
    
    pt = data['pt'][indices]
    assert pt.shape == gamma_t.shape
    return pt

  @staticmethod
  def backward(ctx, grad_output):
    return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    with fsspec.open(self.config.algo.integral_cache_path,
                     'rb') as f:
      self.integral_cache = pickle.load(f)
    self.integral_cache['pt'] = torch.from_numpy(
      self.integral_cache['pt'])
    self.integral_cache['grad_pt'] = torch.from_numpy(
      self.integral_cache['grad_pt'])
    self.gamma_min = self.config.algo.gamma_min
    self.gamma_max = self.config.algo.gamma_max
    self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
    self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
    self.curriculum_start = self.config.algo.curriculum_start
    self.curriculum_end = self.config.algo.curriculum_end
    self.loss_type = self.config.algo.loss_type
    self.is_adversarial_distill = self.config.adversarial_distill.is_distill
    if self.is_adversarial_distill:
      self.automatic_optimization = False
      self.teacher = copy.deepcopy(self.backbone)
      utils.freeze_model(self.teacher)
      
      self.fake_model = copy.deepcopy(self.backbone)
      utils.activate_model(self.fake_model)
      utils.activate_model(self.backbone)
      if self.ema is not None:
        del self.ema
        self.ema = models.ema.ExponentialMovingAverage(
          self.backbone.parameters(),
          decay=self.config.training.ema)

        self.ema_fake = models.ema.ExponentialMovingAverage(
          self.fake_model.parameters(),
          decay=self.config.training.ema)
      self.training_step_counter = 0
      self.accum_batches = self.config.adversarial_distill.accum_batches 
      self._fake_accum_counter = 0
      self._student_accum_counter = 0
      self.fake_frequency = self.config.adversarial_distill.fake_frequency
      self.fake_iter = 0 
      self.is_generator_step = False
      self.total_loss_student = 0
      self.total_loss_fake = 0
      self.f_loss_ar = []
    self._validate_configuration()

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.integral_cache['pt'] = self.integral_cache[
      'pt'].to(*args, **kwargs)
    self.integral_cache['grad_pt'] = self.integral_cache[
      'grad_pt'].to(*args, **kwargs)
    return self

  def _compute_gumbel_tau_inverse(self):
    start = self.gumbel_tau_log10_start
    end = self.gumbel_tau_log10_end
    delta = end - start
    if self.global_step < self.curriculum_start:
      tau = start
    elif self.global_step < self.curriculum_end:
      frac = (self.global_step - self.curriculum_start) / (
        self.curriculum_end - self.curriculum_start)
      tau = start + frac * delta
    else:
      tau = -10
    return 10 ** (-tau)

  def _gaussian_update(self, x, t, dt):
    t = t.squeeze()
    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    sigma = self._sigma_from_alphat(usdm_alpha_t)
    log_x_theta = self.forward(x, sigma=sigma)
    x0 = trainer_base.sample_categorical(log_x_theta.exp())
    x0_one_hot = F.one_hot(x0, self.vocab_size)
    

    gamma_t_mdt = self.gamma_min + (t - dt) * (self.gamma_max - self.gamma_min)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t_mdt)
    return xt

  def _clean_probs(self, x0):
    return F.one_hot(x0, self.vocab_size).to(self.dtype)

  def _force_prompt_probs(self, probs, clean_probs, valid_tokens):
    if valid_tokens is None:
      return probs
    prompt = ~valid_tokens.bool().unsqueeze(-1)
    return torch.where(prompt, clean_probs, probs)

  def _scale_and_restore_prompt(self, xt, clean_probs, valid_tokens):
    tau_inv = self._compute_gumbel_tau_inverse()
    xt = xt * tau_inv
    if valid_tokens is not None:
      prompt = ~valid_tokens.bool().unsqueeze(-1)
      xt = torch.where(prompt, clean_probs * tau_inv, xt)
    return xt

  def _argmax_with_clean_prompt(self, xt, x0, valid_tokens):
    xt_tokens = xt.argmax(-1)
    if valid_tokens is not None:
      xt_tokens = torch.where(valid_tokens.bool(), xt_tokens, x0)
    return xt_tokens

  def _masked_mean(self, loss, valid_tokens):
    if valid_tokens is None:
      return loss.sum(-1).mean()
    return (loss * valid_tokens).sum() / valid_tokens.sum().clamp(min=1)

  def multistep_generation(self, x0, current_accumulation_step,
                           valid_tokens=None, eps_probs=1e-12):
    t = self._sample_t(x0.shape[0], current_accumulation_step)

    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)  

    x0_one_hot = self._clean_probs(x0)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
    xt = self._scale_and_restore_prompt(xt, x0_one_hot, valid_tokens)

    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(usdm_alpha_t)
    
    log_x_theta = self.forward(xt, sigma=sigma)
    student_probs = log_x_theta.exp()
    return self._force_prompt_probs(student_probs, x0_one_hot, valid_tokens)

  def nll_per_token_with_probs(self, log_x_theta, xt, x0, alpha_t, dalpha_t):
    assert alpha_t.ndim == 2
    assert x0.ndim == 3
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[:, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    x_bar = self.vocab_size * alpha_t[:, :, None] * x0 + 1 - alpha_t[:, :, None]

    coeff = dalpha_t / (self.vocab_size * alpha_t)
    
    x_bar_zt = torch.gather(x_bar, -1, xt[..., None])  # B, L, 1
    x_bar_theta_zt = torch.gather(x_bar_theta, -1, xt[..., None])  # B, L, 1
    term1 = ((self.vocab_size / x_bar_zt) - (self.vocab_size / x_bar_theta_zt))  # B, L, 1
    
    term2 = (  # B, L, V before summing --> B, L, 1 after
          (x_bar / x_bar_zt) *
          (
              x_bar_theta_zt.log() - x_bar_theta.log() +
              x_bar.log() - x_bar_zt.log()
          )
      )
    
    term2 = term2.sum(dim=-1, keepdim=True)
    diffusion_loss = (coeff.unsqueeze(-1) * (term1 - term2)).squeeze(-1)
    assert diffusion_loss.ndim == 2
    return diffusion_loss

  def q_xt_with_probs(self, student_probs, alpha_t):
    noise_probs = alpha_t[:, :, None] * student_probs + (1 - alpha_t[:, :, None])*(1/self.vocab_size)
    return noise_probs
  
  def gumbel_softmax_from_probs(self, probs, tau=1e-3, eps=1e-12):
    probs = probs.clamp(min=0.0) + eps
    probs = probs / probs.sum(dim=-1, keepdim=True)

    logits = torch.log(probs)              

    u = torch.rand_like(logits).clamp_(eps, 1 - eps)
    g = -torch.log(-torch.log(u))
    log_y_soft =(logits + g) / tau

    return log_y_soft

  def generator_loss(self, x0, current_accumulation_step, valid_tokens=None):
    student_probs = self.multistep_generation(
      x0, current_accumulation_step, valid_tokens)
    clean_probs = self._clean_probs(x0)
    student_probs = self._force_prompt_probs(
      student_probs, clean_probs, valid_tokens)

    t = self._sample_t(x0.shape[0], current_accumulation_step)
    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)   
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)

    sigma = self._sigma_from_alphat(usdm_alpha_t)
    sigma = self._process_sigma(sigma)

    xt = self._q_xt_gaussian(student_probs, gamma_t)
    xt = self._scale_and_restore_prompt(xt, clean_probs, valid_tokens)
    xt_usdm = self._argmax_with_clean_prompt(xt, x0, valid_tokens)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_teacher = self.teacher(xt, sigma)
    logits_teacher = self._process_model_output(model_output=model_output_teacher, xt=xt, sigma=sigma)
    teacher_loss = self.nll_per_token_with_probs(logits_teacher, xt_usdm, student_probs, 
                                alpha_t=usdm_alpha_t, dalpha_t=usdm_dalpha_t)
    
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_fake = self.fake_model(xt, sigma)
    logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
    fake_loss = self.nll_per_token_with_probs(logits_fake, xt_usdm, student_probs, 
                                alpha_t=usdm_alpha_t, dalpha_t=usdm_dalpha_t)
    return self._masked_mean(teacher_loss - fake_loss, valid_tokens)
  
  def configure_optimizers(self):
    if self.config.adversarial_distill.is_distill:
      optimizer_student = torch.optim.AdamW(
        self.backbone.parameters(),
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
              self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)
      
      optimizer_fake = torch.optim.AdamW(
        self.fake_model.parameters(),
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
              self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)

      scheduler_student = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer_student)
      scheduler_student_dict = {'scheduler': scheduler_student, 'name': 'trainer/lr'}

      scheduler_fake = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer_fake)
      scheduler_fake_dict = {'scheduler': scheduler_fake, 'name': 'trainer/lr_fake'}
      return [optimizer_student, optimizer_fake], [scheduler_student_dict, scheduler_fake_dict]
    else:
      return super().configure_optimizers()

  def fake_loss(self, x0, current_accumulation_step, valid_tokens=None):
    with torch.no_grad():
      student_probs = self.multistep_generation(
        x0, current_accumulation_step, valid_tokens)
    clean_probs = self._clean_probs(x0)
    student_probs = self._force_prompt_probs(
      student_probs, clean_probs, valid_tokens)

    t = self._sample_t(x0.shape[0], current_accumulation_step)

    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)   
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)

    sigma = self._sigma_from_alphat(usdm_alpha_t)
    sigma = self._process_sigma(sigma)

    xt = self._q_xt_gaussian(student_probs, gamma_t)
    xt = self._scale_and_restore_prompt(xt, clean_probs, valid_tokens)
    xt_usdm = self._argmax_with_clean_prompt(xt, x0, valid_tokens)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_fake = self.fake_model(xt, sigma)
    logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
    fake_loss = self.nll_per_token_with_probs(logits_fake, xt_usdm, student_probs, 
                                alpha_t=usdm_alpha_t, dalpha_t=usdm_dalpha_t)
    return self._masked_mean(fake_loss, valid_tokens)

  def _apply_checkpoint_ema_to_backbone_state(self, state_dict, checkpoint):
    ema_state = checkpoint.get('ema', None)
    if not (ema_state and 'shadow_params' in ema_state):
      return
    for (name, _), shadow in zip(
        self.backbone.named_parameters(), ema_state['shadow_params']):
      key = f'backbone.{name}'
      if key in state_dict:
        state_dict[key] = shadow.detach().clone().to(state_dict[key].dtype)

  def _backbone_shadow_params_from_state(self, state_dict):
    shadow_params = []
    for name, _ in self.backbone.named_parameters():
      key = f'backbone.{name}'
      if key not in state_dict:
        return None
      shadow_params.append(state_dict[key].detach().clone())
    return shadow_params

  def _reset_distill_ema(self, shadow_params):
    if shadow_params is None:
      return
    ema_state = {
      'decay': self.config.training.ema,
      'num_updates': 0,
      'shadow_params': shadow_params,
    }
    if self.ema is not None:
      self.ema.load_state_dict(copy.deepcopy(ema_state))
    if getattr(self, 'ema_fake', None) is not None:
      self.ema_fake.load_state_dict(copy.deepcopy(ema_state))

  def on_train_start(self):
    if self.is_adversarial_distill:
      self.ema_fake.move_shadow_params_to_device(self.device)
      utils.activate_model(self.fake_model)
      utils.freeze_model(self.backbone)
    super().on_train_start()

  def on_load_checkpoint(self, checkpoint):
    state_dict = collections.OrderedDict(checkpoint['state_dict'])
    has_distill_state = any(
      key.startswith(('teacher.', 'fake_model.'))
      for key in state_dict)
    if self.is_adversarial_distill:
      if not has_distill_state:
        self._apply_checkpoint_ema_to_backbone_state(state_dict, checkpoint)
      for key, value in list(state_dict.items()):
        if key.startswith('backbone.'):
          suffix = key[len('backbone.'):]
          state_dict.setdefault(f'teacher.{suffix}', value)
          state_dict.setdefault(f'fake_model.{suffix}', value)
    else:
      state_dict = collections.OrderedDict(
        (k, v) for k, v in state_dict.items()
        if not k.startswith(('teacher.', 'fake_model.')))
    checkpoint['state_dict'] = state_dict
    if self.is_adversarial_distill and not has_distill_state:
      self._reset_distill_ema(
        self._backbone_shadow_params_from_state(state_dict))
      return
    trainer_base.TrainerBase.on_load_checkpoint(self, checkpoint)
    if (self.is_adversarial_distill
        and getattr(self, 'ema_fake', None) is not None):
      ema_fake_state = checkpoint.get('ema_fake', checkpoint.get('ema'))
      if ema_fake_state is not None:
        self.ema_fake.load_state_dict(ema_fake_state)

  def on_save_checkpoint(self, checkpoint):
    if self.is_adversarial_distill:
      checkpoint['state_dict'] = collections.OrderedDict(
        (k, v) for k, v in checkpoint['state_dict'].items()
        if not k.startswith('teacher.'))
      trainer_base.TrainerBase.on_save_checkpoint(self, checkpoint)
    else:
      super().on_save_checkpoint(checkpoint)
    if (self.is_adversarial_distill
        and getattr(self, 'ema_fake', None) is not None):
      checkpoint['ema_fake'] = self.ema_fake.state_dict()

  def training_step(self, batch, batch_idx):
    self.log(name='gumbel_tau_log10',
              value=1 / self._compute_gumbel_tau_inverse(),
              on_step=True,
              on_epoch=False,
              sync_dist=True)
    if self.config.adversarial_distill.is_distill:
      optimizer_student, optimizer_fake = self.optimizers()
      scheduler_student, scheduler_fake = self.lr_schedulers()
      current_accumulation_step = (batch_idx % self.accum_batches)
      finish_iter = False

      (input_tokens, output_tokens, valid_tokens) = self._process_model_input(batch['input_ids'], 
                                                                              batch['attention_mask'])
      valid_tokens = _conditional_valid_tokens(self.config, valid_tokens)
      if self.is_generator_step:
        self.toggle_optimizer(optimizer_student)
        g_loss = self.generator_loss(
          input_tokens, current_accumulation_step, valid_tokens)
        g_loss = g_loss.mean()
        g_loss = g_loss / self.accum_batches
        self.manual_backward(g_loss)

        self.untoggle_optimizer(optimizer_student)

        self._student_accum_counter += 1
        self.total_loss_student += g_loss.detach()
        if self._student_accum_counter == self.accum_batches:
          self._student_accum_counter = 0 
          optimizer_student.step()
          self.ema.update(self.backbone.parameters())
          optimizer_student.zero_grad()
          finish_iter = True
      else:
        self.toggle_optimizer(optimizer_fake)
        f_loss = self.fake_loss(
          input_tokens, current_accumulation_step, valid_tokens)
        f_loss = f_loss.mean()
        f_loss = f_loss / self.accum_batches
        self.manual_backward(f_loss)

        self.untoggle_optimizer(optimizer_fake)
        self._fake_accum_counter += 1
        self.total_loss_fake += f_loss.detach()
        if self._fake_accum_counter == self.accum_batches:
          self._fake_accum_counter = 0 
          optimizer_fake.step()
          self.ema_fake.update(self.fake_model.parameters())
          optimizer_fake.zero_grad()
          finish_iter = True

      if finish_iter:
        _maybe_save_periodic_checkpoint(self)

      if finish_iter and self.is_generator_step:
        self.is_generator_step = False
        utils.activate_model(self.fake_model)
        utils.freeze_model(self.backbone)
        optimizer_fake.zero_grad()
        self.log_dict(
            {"student_loss": self.total_loss_student, "fake_loss": torch.stack(self.f_loss_ar).mean()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        scheduler_student.step()
        scheduler_fake.step()
        self.total_loss_student = 0
        self.f_loss_ar.clear()
      elif finish_iter:
        self.fake_iter += 1
        self.f_loss_ar.append(self.total_loss_fake)
        self.total_loss_fake = 0
        if self.fake_iter == self.fake_frequency:
          self.is_generator_step = True
          self.fake_iter = 0
          utils.activate_model(self.backbone)
          utils.freeze_model(self.fake_model)
          optimizer_student.zero_grad()

    else:
      return super().training_step(batch, batch_idx)

  def _gamma_to_alphat(self, gamma_t):
    integral = Integral.apply(gamma_t, self.integral_cache)
    return (self.vocab_size * integral - 1) / (
      self.vocab_size - 1)

  def _prior_loss(self):
    alpha_1 = self._gamma_to_alphat(
      torch.tensor(self.gamma_max))
    loss = ((alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
      (self.vocab_size - 1) * alpha_1 + 1) + (
        1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1))
    return loss.item()

  def _q_xt_gaussian(self, x, gamma_t):
    """Computes the noisy sample xt."""
    assert gamma_t.ndim == 1
    assert x.ndim == 3
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()
    epsilon = torch.randn(x.shape, dtype=torch.float32,
                          device=self.device)
    return alpha_t * x + sigma_t * epsilon

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    use_true_nll = (self.global_step > self.curriculum_end
                    or not train_mode)
    if use_true_nll:
      return super().nll(x0, output_tokens,
                         current_accumulation_step,
                         valid_tokens=valid_tokens)
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)    
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (
      self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(usdm_alpha_t)

    x0_one_hot = F.one_hot(x0, self.vocab_size)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
    xt = xt * self._compute_gumbel_tau_inverse()
    xt_usdm = xt.argmax(-1)
    log_x_theta = self.forward(xt, sigma=sigma)

    return self.nll_per_token(log_x_theta=log_x_theta,
                              xt=xt_usdm,
                              x0=x0,
                              alpha_t=usdm_alpha_t,
                              dalpha_t=usdm_dalpha_t,
                              low_var=False)


class Distillation(DUO):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.update_teacher_every = config.algo.update_teacher_every
    self.save_hyperparameters()
    self.teacher = None
    self.teacher_ema = config.algo.teacher_ema
    self.linear_growth_dt = config.algo.linear_growth_dt
    self.linear_growth_min = config.algo.linear_growth_min
    self.linear_growth_max = config.algo.linear_growth_max

  def _validate_configuration(self):
    assert os.path.exists(
      self.config.algo.integral_cache_path), (
        'The integral cache (Eq. 10 in the paper) for '
        f'the {self.config.data.tokenizer_name_or_path} '
        ' tokenizer doesnt exist at '
        f'{self.config.algo.integral_cache_path}. '
        'Please generate it by running the utils.py script, '
        'and ensure the correct path is specified using the '
        'algo.integral_cache_path flag.')
    assert self.loss_type in {
      'kl-fwd', 'kl-bwd', 'posterior', 'kl-posterior'}

  def _maybe_update_teacher_weights(self):
    if self.global_step % self.update_teacher_every != 0:
      return
    if self.teacher_ema:
      self.ema.copy_to(self.teacher.parameters())
    else:
      for better_param, current_param in zip(
        self.backbone.parameters(), self.teacher.parameters()):
        if current_param.requires_grad:
          current_param.data.copy_(better_param.data)

  @torch.no_grad()
  def _teacher_logits(self, xt, sigma):
    if self.teacher is None:
      self.teacher = copy.deepcopy(self.backbone)
    self._maybe_update_teacher_weights()

    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output = self.teacher(xt, sigma)
    logits = self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)
    return logits.detach()

  def _sample_trajectory(self, x0, gamma_t, gamma_s):
    """Computes the noisy sample xt."""
    assert gamma_t.ndim == 1
    assert gamma_s.ndim == 1
    assert x0.ndim == 2
    x0 = F.one_hot(x0, self.vocab_size).to(
      self.dtype).to(self.device)
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()

    gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
    alpha_s = torch.sigmoid(-gamma_s).sqrt()
    sigma_s = torch.sigmoid(gamma_s).sqrt()
    
    epsilon = torch.randn(x0.shape, dtype=torch.float32,
                          device=self.device)
    xt = alpha_t * x0 + sigma_t * epsilon
    xs = alpha_s * x0 + sigma_s * epsilon
    return xt, xs

  def _compute_dt(self):
    if self.linear_growth_dt:
      scale = self.global_step / self.trainer.max_steps
      return self.linear_growth_min + scale * (
        self.linear_growth_max -  self.linear_growth_min)
    n = self.global_step // self.update_teacher_every
    return 2 ** n / self.T

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=None,
          valid_tokens=None):
    del output_tokens, train_mode
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dt = self._compute_dt()
    t = torch.clip(t + dt, 0, 1)

    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)
    gamma_s = self.gamma_min + (
      t - dt) * (self.gamma_max - self.gamma_min)

    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    usdm_alpha_s = self._gamma_to_alphat(gamma_s)
    usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
    assert usdm_alpha_s.ndim == 2

    xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
    xt_discrete = xt.argmax(-1)
    xs_discrete = xs.argmax(-1)
    log_x_theta_student = self.forward(
      xt_discrete, sigma=self._sigma_from_alphat(usdm_alpha_t))
    log_x_theta_teacher = self._teacher_logits(
      xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s))
    if self.config.training.loss_precision == 'float64':
      log_x_theta_student = log_x_theta_student.to(torch.float64)
      log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
    if self.loss_type == 'kl-fwd':
      return (log_x_theta_teacher.exp() * (
        log_x_theta_teacher - log_x_theta_student)).sum(-1)
    elif self.loss_type == 'kl-bwd':
      return (log_x_theta_student.exp() * (
        log_x_theta_student - log_x_theta_teacher)).sum(-1)
    
  def training_step(self, batch, batch_idx):
    self.log(name='dt',
             value=self._compute_dt(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return super().training_step(batch, batch_idx)

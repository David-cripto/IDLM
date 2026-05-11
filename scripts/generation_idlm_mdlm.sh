noise_removal=ancestral

for steps in 4 8 16 32; do
  echo "Running with steps = $steps"

  python main.py \
    mode=sample_eval \
    loader.batch_size=2 \
    loader.eval_batch_size=8 \
    data=openwebtext-split \
    algo=mdlm \
    algo.backbone='hf_dit' \
    eval.checkpoint_path="kekchpek/idlm-mdlm" \
    sampling.steps="$steps" \
    sampling.num_sample_batches=10 \
    sampling.noise_removal=$noise_removal \
    +wandb.offline=true \
    sampling.predictor='ancestral_cache' \
    eval.generated_samples_path=path-to-save-dir
done
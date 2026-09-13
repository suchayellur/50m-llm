param(
  [string]$ModelDir = "runs/final/final",
  [string]$OutputDir = "eval_results"
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
lm_eval `
  --model hf `
  --model_args "pretrained=$ModelDir,trust_remote_code=True" `
  --tasks hellaswag,arc_easy,piqa,winogrande,wikitext `
  --device cuda:0 `
  --batch_size auto `
  --output_path $OutputDir

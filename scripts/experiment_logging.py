import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def software_versions():
    packages = ["torch", "transformers", "datasets", "tokenizers", "accelerate", "lm-eval"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_markdown(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Run Report",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Git commit: `{payload.get('git_commit') or 'not available'}`",
        f"- Started: `{payload.get('training', {}).get('start_time_utc')}`",
        f"- Ended: `{payload.get('training', {}).get('end_time_utc')}`",
        f"- Duration seconds: `{payload.get('training', {}).get('duration_seconds')}`",
        f"- Final training loss: `{payload.get('training', {}).get('final_loss')}`",
        "",
        "## Model",
        "",
        f"- Trainable parameters: `{payload.get('model', {}).get('trainable_parameters')}`",
        f"- Parameter limit: `{payload.get('model', {}).get('parameter_limit')}`",
        f"- Random initialization: `{payload.get('model', {}).get('random_initialization')}`",
        f"- Pretrained initialization used: `{payload.get('model', {}).get('pretrained_initialization_used')}`",
        "",
        "## Data",
        "",
        f"- Dataset: `{payload.get('data', {}).get('dataset')}`",
        f"- Dataset version/revision: `{payload.get('data', {}).get('dataset_version')}`",
        f"- Dataset mixture: `{payload.get('data', {}).get('dataset_mixture')}`",
        f"- Local files: `{payload.get('data', {}).get('local_files')}`",
        "",
        "## Training",
        "",
        f"- Seed: `{payload.get('training', {}).get('seed')}`",
        f"- Total tokens: `{payload.get('training', {}).get('total_training_tokens')}`",
        f"- Optimizer: `{payload.get('training', {}).get('optimizer')}`",
        f"- Learning rate: `{payload.get('training', {}).get('learning_rate')}`",
        f"- Batch size: `{payload.get('training', {}).get('batch_size')}`",
        f"- Gradient accumulation: `{payload.get('training', {}).get('gradient_accumulation')}`",
        f"- Sequence length: `{payload.get('training', {}).get('sequence_length')}`",
        f"- Precision: `{payload.get('hardware', {}).get('precision')}`",
        "",
        "## Hardware",
        "",
        f"- GPU name(s): `{payload.get('hardware', {}).get('gpu_names')}`",
        f"- GPU count: `{payload.get('hardware', {}).get('gpu_count')}`",
        f"- Peak GPU memory bytes: `{payload.get('hardware', {}).get('peak_gpu_memory_bytes')}`",
        "",
        "## Checkpoints",
        "",
    ]
    checkpoints = payload.get("training", {}).get("checkpoints", [])
    if checkpoints:
        lines.extend(f"- `{item}`" for item in checkpoints)
    else:
        lines.append("- None")
    lines.extend(["", "## Software", "", "```json", json.dumps(payload.get("software", {}), indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

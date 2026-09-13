import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import HackathonLMConfig, HackathonLMForCausalLM


LIMIT = 50_000_000


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Instantiate a random model and count trainable parameters.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = HackathonLMConfig.from_json_file(str(config_path))
    model = HackathonLMForCausalLM(config)
    total = count_trainable(model)

    print(json.dumps(config.to_dict(), indent=2))
    print(f"Trainable parameters: {total:,}")
    print(f"Parameter limit:       {LIMIT:,}")
    assert total <= LIMIT, f"Model has {total:,} trainable parameters, above {LIMIT:,}"


if __name__ == "__main__":
    main()

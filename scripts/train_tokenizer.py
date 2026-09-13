import argparse
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from data.datasets import choose_text_stream


def main():
    parser = argparse.ArgumentParser(description="Train a tokenizer from scratch.")
    parser.add_argument("--dataset", choices=["tinystories", "fineweb", "local"], default="tinystories")
    parser.add_argument("--local-files", nargs="*", default=[])
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--limit-texts", type=int, default=200_000)
    parser.add_argument("--output-dir", default="tokenizer")
    args = parser.parse_args()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)

    texts = islice(choose_text_stream(args.dataset, args.local_files), args.limit_texts)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=args.limit_texts)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(output_dir)
    print(f"Saved {args.vocab_size}-token tokenizer to {output_dir}")


if __name__ == "__main__":
    main()

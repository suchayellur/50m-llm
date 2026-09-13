import argparse
import random
from pathlib import Path


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?-"


def caesar(text, shift):
    out = []
    for ch in text:
        if ch in ALPHABET:
            out.append(ALPHABET[(ALPHABET.index(ch) + shift) % len(ALPHABET)])
        else:
            out.append(ch)
    return "".join(out)


def xor_hex(text, key):
    return bytes([b ^ key for b in text.encode("utf-8")]).hex()


def make_example(rng):
    plain = " ".join(rng.choice([
        "meet at dawn",
        "the package is hidden",
        "rotate the key",
        "language models learn patterns",
        "cryptography needs careful tests",
    ]) for _ in range(rng.randint(1, 3)))
    if rng.random() < 0.5:
        shift = rng.randint(1, 25)
        cipher = caesar(plain, shift)
        return f"Task: decrypt caesar shift {shift}\nCiphertext: {cipher}\nPlaintext: {plain}\n"
    key = rng.randint(1, 255)
    cipher = xor_hex(plain, key)
    return f"Task: decrypt xor hex key {key}\nCiphertext: {cipher}\nPlaintext: {plain}\n"


def main():
    parser = argparse.ArgumentParser(description="Create small procedural cryptography text data.")
    parser.add_argument("--output", default="crypto_train.txt")
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for _ in range(args.examples):
            f.write(make_example(rng) + "\n")
    print(f"Wrote {args.examples} examples to {output}")


if __name__ == "__main__":
    main()

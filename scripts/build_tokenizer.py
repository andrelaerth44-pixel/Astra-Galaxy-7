import argparse
from pathlib import Path

from src.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Astra Galaxy 7 BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Clean text files.")
    parser.add_argument("--output", required=True, help="Output tokenizer.json path.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    train_tokenizer(
        [Path(p) for p in args.input],
        args.output,
        vocab_size=args.vocab_size,
    )
    print(f"tokenizer written to {args.output}")


if __name__ == "__main__":
    main()

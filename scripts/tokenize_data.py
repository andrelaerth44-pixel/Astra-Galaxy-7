import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Encode text into a compact uint32 token stream.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    text = Path(args.input).read_text(encoding="utf-8")
    ids = tokenizer.encode(text).ids

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=np.uint32).tofile(output)
    print(f"encoded {len(ids):,} tokens -> {output}")


if __name__ == "__main__":
    main()

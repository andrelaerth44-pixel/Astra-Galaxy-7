import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare JSONL text while preserving formatting by default.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--normalize-whitespace",
        action="store_true",
        help="Collapse whitespace. Do not use this for source code corpora.",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    chars = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str):
                continue
            if args.normalize_whitespace:
                text = " ".join(text.split())
            else:
                text = text.strip()
            if not text:
                continue
            fout.write(text + "\n")
            count += 1
            chars += len(text)

    print(f"prepared {count:,} documents and {chars:,} characters -> {dst}")


if __name__ == "__main__":
    main()

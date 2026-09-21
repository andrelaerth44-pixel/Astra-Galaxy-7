import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Clean JSONL text documents.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    chars = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str):
                continue
            text = " ".join(text.split())
            if not text:
                continue
            fout.write(text + "\n")
            count += 1
            chars += len(text)

    print(f"prepared {count:,} documents and {chars:,} characters -> {dst}")

if __name__ == "__main__":
    main()

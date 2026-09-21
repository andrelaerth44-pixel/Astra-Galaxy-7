import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            fout.write(text.replace("\n", "
").strip() + "
")
            count += 1

    print(f"prepared {count} documents -> {dst}")


if __name__ == "__main__":
    main()

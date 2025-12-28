import argparse
import glob
import json
import os
from typing import Generator, Iterable, List, Union


JsonValue = Union[dict, list, str, int, float, bool, None]


def iter_json_items_from_file(file_path: str) -> Generator[JsonValue, None, None]:
    """Yield items from a JSON file.

    Supports two formats:
    - Standard JSON array: [obj, obj, ...]
    - Newline-delimited JSON (NDJSON/JSONL): one JSON object per line
    If the top-level JSON is a dict, it yields that dict as a single item.
    """
    # Try whole-file JSON first
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
            return
        if isinstance(data, dict):
            # If there is a likely list container, unwrap it; otherwise yield the dict itself
            for key in ("items", "data", "articles", "records", "results"):  # common wrappers
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        yield item
                    return
            yield data
            return
        # Primitive (str/number/bool/null): yield as-is
        yield data
        return
    except json.JSONDecodeError:
        # Fallback to NDJSON/JSONL: parse line by line
        pass

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                # Skip malformed lines but keep going
                continue


def merge_json_files(input_glob: str, output_path: str) -> int:
    """Merge all matching JSON files into a single JSON array file.

    Returns the number of items written.
    """
    matching_files = sorted(glob.glob(input_glob))
    if not matching_files:
        raise FileNotFoundError(f"No files matched pattern: {input_glob}")

    total_items = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write as a JSON array without keeping everything in memory at once
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("[")
        first = True
        for file_path in matching_files:
            for item in iter_json_items_from_file(file_path):
                if first:
                    first = False
                else:
                    out.write(",\n")
                json.dump(item, out, ensure_ascii=False)
                total_items += 1
        out.write("]\n")

    return total_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge JSON files into one JSON array.")
    parser.add_argument(
        "--pattern",
        default="vnexpress_*.json",
        help="Glob pattern for input files (default: vnexpress_*.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="merged.json",
        help="Output JSON file path (default: merged.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = merge_json_files(args.pattern, args.output)
    print(f"Merged {count} items into {args.output}")


if __name__ == "__main__":
    main()




import random
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    dataset = load_dataset("CarperAI/pickapic_v1_no_images_training_sfw", num_proc=16)

    text_dataset = dataset["train"].select_columns(["caption"])
    unique_dataset = text_dataset.unique("caption")
    unique_dataset = [s for s in unique_dataset if s.count(" ") >= 5]
    random.shuffle(unique_dataset)

    test_size = 1024
    test_dataset = unique_dataset[:test_size]
    train_dataset = unique_dataset[test_size:]

    _write_lines(OUTPUT_DIR / "train.txt", train_dataset)
    _write_lines(OUTPUT_DIR / "test.txt", test_dataset)


def _write_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


if __name__ == "__main__":
    main()

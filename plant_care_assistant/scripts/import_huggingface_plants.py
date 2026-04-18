"""Import selected plant classes from a Hugging Face image dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != PROJECT_ROOT]

from datasets import load_dataset  # noqa: E402


DEFAULT_LABEL_MAP = {
    "Aloevera": "Aloe Vera",
    "Hibiscus": "Hibiscus",
    "Rose": "Rose",
    "Tulasi": "Tulsi",
}


def _get_label_name(example: dict, label_names: list[str]) -> str:
    label = example["label"]
    if isinstance(label, int):
        return label_names[label]
    return str(label)


def import_huggingface_plants(
    dataset_name: str,
    target_dir: Path,
    max_images_per_class: int,
    split: str,
) -> None:
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    label_feature = dataset.features["label"]
    label_names = list(getattr(label_feature, "names", []))
    copied_counts = {plant_name: 0 for plant_name in DEFAULT_LABEL_MAP.values()}

    target_dir.mkdir(parents=True, exist_ok=True)

    for index, example in enumerate(dataset):
        source_label = _get_label_name(example, label_names)
        plant_name = DEFAULT_LABEL_MAP.get(source_label)
        if plant_name is None:
            continue
        if copied_counts[plant_name] >= max_images_per_class:
            if all(count >= max_images_per_class for count in copied_counts.values()):
                break
            continue

        image = example["image"].convert("RGB")
        plant_dir = target_dir / plant_name
        plant_dir.mkdir(parents=True, exist_ok=True)
        output_path = plant_dir / f"{dataset_name.replace('/', '_')}_{source_label}_{index}.jpg"
        image.save(output_path, quality=90)
        copied_counts[plant_name] += 1

    print(f"Imported selected images from {dataset_name} into {target_dir}")
    for plant_name, count in sorted(copied_counts.items()):
        print(f"- {plant_name}: {count} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import selected Hugging Face plant images.")
    parser.add_argument(
        "--dataset",
        default="funkepal/medicinal_plant_images",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=PROJECT_ROOT / "training_images",
        help="Output training_images folder.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=160,
        help="Maximum images to import for each mapped plant class.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to import.")
    args = parser.parse_args()

    import_huggingface_plants(
        dataset_name=args.dataset,
        target_dir=args.target_dir,
        max_images_per_class=args.max_images_per_class,
        split=args.split,
    )


if __name__ == "__main__":
    main()

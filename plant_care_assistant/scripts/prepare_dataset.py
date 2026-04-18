"""Prepare local plant image folders for classifier training.

This script works with downloaded/extracted image datasets. It copies matching
classes into training_images/<Plant Name>/ so train_image_model.py can use them.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLANT_ALIASES = {
    "Aloe Vera": ["aloe vera", "aloe"],
    "Aglaonema": ["aglaonema", "chinese evergreen"],
    "Anthurium": ["anthurium"],
    "Areca Palm": ["areca palm", "areca"],
    "Hibiscus": ["hibiscus"],
    "Jade Plant": ["jade plant", "crassula ovata"],
    "Money Plant": ["money plant", "pothos", "golden pothos", "devil's ivy", "devils ivy"],
    "Patharchatta": ["patharchatta", "kalanchoe", "bryophyllum"],
    "Peace Lily": ["peace lily", "spathiphyllum"],
    "Rose": ["rose"],
    "Satavar": ["satavar", "shatavari", "asparagus racemosus"],
    "Snake Plant": ["snake plant", "sansevieria", "dracaena trifasciata"],
    "Spider Plant": ["spider plant", "chlorophytum comosum"],
    "Syngonium": ["syngonium", "arrowhead plant"],
    "Tulsi": ["tulsi", "holy basil", "ocimum tenuiflorum"],
}


def _normalize(text: str) -> str:
    return " ".join(
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
        .split()
    )


def _find_image_folders(source_dir: Path) -> list[Path]:
    image_folders = set()
    for image_path in source_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_folders.add(image_path.parent)
    return sorted(image_folders)


def _match_plant(folder: Path) -> str | None:
    folder_text = _normalize(folder.name)
    parent_text = _normalize(folder.parent.name)
    combined_text = f"{parent_text} {folder_text}"

    for plant_name, aliases in PLANT_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize(alias)
            if normalized_alias in folder_text or normalized_alias in combined_text:
                return plant_name
    return None


def prepare_dataset(source_dir: Path, target_dir: Path, max_images_per_class: int) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {source_dir}. Download and extract the dataset first."
        )
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Expected an extracted dataset folder, got: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    copied_counts = {plant_name: 0 for plant_name in PLANT_ALIASES}
    unmatched_folders = []

    for folder in _find_image_folders(source_dir):
        plant_name = _match_plant(folder)
        if plant_name is None:
            unmatched_folders.append(folder)
            continue

        destination = target_dir / plant_name
        destination.mkdir(parents=True, exist_ok=True)

        image_paths = [
            path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image_path in image_paths:
            if copied_counts[plant_name] >= max_images_per_class:
                break
            output_name = f"{folder.name}_{image_path.name}".replace(" ", "_")
            shutil.copy2(image_path, destination / output_name)
            copied_counts[plant_name] += 1

    print(f"Prepared training images in {target_dir}")
    for plant_name, count in sorted(copied_counts.items()):
        if count:
            print(f"- {plant_name}: {count} images")

    missing = [plant_name for plant_name, count in sorted(copied_counts.items()) if count == 0]
    if missing:
        print("\nNo matching images found for:")
        print(", ".join(missing))

    if unmatched_folders:
        print("\nSome dataset folders were not matched to app plant names.")
        print("First unmatched folders:")
        for folder in unmatched_folders[:15]:
            print(f"- {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a plant dataset for training.")
    parser.add_argument("source_dir", type=Path, help="Downloaded/extracted dataset folder.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "training_images",
        help="Output training_images folder.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=80,
        help="Maximum images to copy for each plant class.",
    )
    args = parser.parse_args()

    prepare_dataset(args.source_dir, args.target_dir, args.max_images_per_class)


if __name__ == "__main__":
    main()

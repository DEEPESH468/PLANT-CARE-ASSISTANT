"""Train the plant image classifier from local training image folders."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from image_classifier import MODEL_PATH, PlantImageClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_features(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classifier = PlantImageClassifier()
    features = []
    labels = []

    for plant_dir in sorted(data_dir.iterdir()):
        if not plant_dir.is_dir():
            continue

        plant_name = plant_dir.name
        for image_path in sorted(plant_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                normalized_image = classifier.preprocess_image(str(image_path))
                features.append(classifier.extract_features(normalized_image))
                labels.append(plant_name)
            except (FileNotFoundError, ValueError) as error:
                print(f"Skipped {image_path}: {error}")

    if not features:
        raise ValueError(f"No training images found in {data_dir}")

    return np.array(features, dtype=np.float32), np.array(labels), sorted(set(labels))


def train_model(data_dir: Path, model_path: Path) -> None:
    features, labels, class_names = _collect_features(data_dir)
    label_counts = Counter(labels)

    if len(class_names) < 2:
        raise ValueError("Training needs at least two plant classes.")

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"),
    )

    can_split = len(features) >= 10 and min(label_counts.values()) >= 2
    if can_split:
        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        print(f"Validation accuracy: {accuracy_score(test_labels, predictions):.2%}")

    model.fit(features, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": labels,
            "class_names": class_names,
        },
        model_path,
    )

    print(f"Trained on {len(features)} images across {len(class_names)} plant classes.")
    for plant_name, count in sorted(label_counts.items()):
        print(f"- {plant_name}: {count} images")
    print(f"Saved model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the plant image classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "training_images",
        help="Folder containing one subfolder per plant class.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Output path for the trained model.",
    )
    args = parser.parse_args()

    train_model(args.data_dir, args.model_path)


if __name__ == "__main__":
    main()

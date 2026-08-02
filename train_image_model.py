"""Train the plant image classifier from local training image folders."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from image_classifier import MODEL_PATH, PlantImageClassifier
from image_classifier import is_likely_correct_training_image, is_supported_image_file


def _collect_training_images(data_dir: Path) -> tuple[list[Path], np.ndarray, list[str], Counter]:
    image_paths = []
    labels = []
    skipped_mismatches = Counter()

    for plant_dir in sorted(data_dir.iterdir()):
        if not plant_dir.is_dir():
            continue

        plant_name = plant_dir.name
        for image_path in sorted(plant_dir.iterdir()):
            if not is_supported_image_file(image_path):
                continue
            if not is_likely_correct_training_image(plant_name, image_path):
                skipped_mismatches[plant_name] += 1
                continue
            image_paths.append(image_path)
            labels.append(plant_name)

    if not image_paths:
        raise ValueError(f"No training images found in {data_dir}")

    for plant_name, count in sorted(skipped_mismatches.items()):
        print(f"Skipped {count} likely mismatched images for {plant_name}.")

    return image_paths, np.array(labels), sorted(set(labels)), skipped_mismatches


def _training_image_variants(normalized_image: np.ndarray) -> list[np.ndarray]:
    """Return deterministic augmentations used only for the training split."""
    crop_size = 112
    crop_start = (normalized_image.shape[0] - crop_size) // 2
    crop_end = crop_start + crop_size
    crop = normalized_image[crop_start:crop_end, crop_start:crop_end]

    return [
        normalized_image,
        np.ascontiguousarray(normalized_image[:, ::-1, :]),
        cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA),
    ]


def _collect_features(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classifier = PlantImageClassifier()
    image_paths, raw_labels, class_names, _ = _collect_training_images(data_dir)
    features = []
    labels = []

    for image_path, plant_name in zip(image_paths, raw_labels):
        try:
            normalized_image = classifier.preprocess_image(str(image_path))
            features.append(classifier.extract_features(normalized_image))
            labels.append(plant_name)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipped {image_path}: {error}")

    return np.array(features, dtype=np.float32), np.array(labels), sorted(set(labels))


def _extract_features_for_indices(
    classifier: PlantImageClassifier,
    image_paths: list[Path],
    labels: np.ndarray,
    indices: np.ndarray,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    output_labels = []

    for image_index in indices:
        image_path = image_paths[int(image_index)]
        plant_name = labels[int(image_index)]
        try:
            normalized_image = classifier.preprocess_image(str(image_path))
            image_variants = _training_image_variants(normalized_image) if augment else [normalized_image]
            for image_variant in image_variants:
                features.append(classifier.extract_features(image_variant))
                output_labels.append(plant_name)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipped {image_path}: {error}")

    return np.array(features, dtype=np.float32), np.array(output_labels)


def _predict_with_image_variants(
    classifier: PlantImageClassifier,
    model,
    image_paths: list[Path],
    indices: np.ndarray,
) -> np.ndarray:
    predictions = []

    for image_index in indices:
        image_path = image_paths[int(image_index)]
        normalized_image = classifier.preprocess_image(str(image_path))
        features = np.array(
            [
                classifier.extract_features(image_variant)
                for image_variant in classifier._prediction_image_variants(normalized_image)
            ],
            dtype=np.float32,
        )
        probabilities = model.predict_proba(features).mean(axis=0)
        predictions.append(model.classes_[int(np.argmax(probabilities))])

    return np.array(predictions)


def train_model(data_dir: Path, model_path: Path) -> None:
    classifier = PlantImageClassifier()
    image_paths, labels, class_names, _ = _collect_training_images(data_dir)
    label_counts = Counter(labels)

    if len(class_names) < 2:
        raise ValueError("Training needs at least two plant classes.")

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"),
    )

    image_indices = np.arange(len(labels))
    can_split = len(image_indices) >= 10 and min(label_counts.values()) >= 2
    if can_split:
        train_indices, test_indices, train_labels, test_labels = train_test_split(
            image_indices,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        train_features, train_labels = _extract_features_for_indices(
            classifier,
            image_paths,
            labels,
            train_indices,
            augment=True,
        )
        test_labels = labels[test_indices]
        model.fit(train_features, train_labels)
        predictions = _predict_with_image_variants(classifier, model, image_paths, test_indices)
        print(f"Validation accuracy: {accuracy_score(test_labels, predictions):.2%}")

    all_features, all_labels = _extract_features_for_indices(
        classifier,
        image_paths,
        labels,
        image_indices,
        augment=True,
    )
    model.fit(all_features, all_labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": all_labels,
            "class_names": class_names,
        },
        model_path,
    )

    print(
        f"Trained on {len(image_paths)} images "
        f"({len(all_labels)} augmented samples) across {len(class_names)} plant classes."
    )
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

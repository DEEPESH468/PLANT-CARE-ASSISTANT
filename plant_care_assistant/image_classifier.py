"""Image preprocessing and lightweight plant image classification module."""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier


TRAINING_IMAGE_DIR = Path(__file__).resolve().parent / "training_images"
MODEL_PATH = Path(__file__).resolve().parent / "plant_image_model.joblib"


class PlantImageClassifier:
    """Simple image classifier trained on synthetic plant feature profiles.

    This keeps the university project self-contained and fully runnable without
    needing a separate labeled image dataset file.
    """

    def __init__(self) -> None:
        self.class_names = [
            "Aloe Vera",
            "Tulsi",
            "Rose",
            "Snake Plant",
            "Money Plant",
            "Peace Lily",
            "Spider Plant",
            "Areca Palm",
            "Jade Plant",
            "Hibiscus",
            "Syngonium",
            "Aglaonema",
            "Anthurium",
            "Satavar",
            "Patharchatta",
        ]
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.training_labels = np.array([])
        self.training_sample_count = 0
        self._train_model()

    def _train_model(self) -> None:
        """Load a saved model, train from real image folders, or use prototypes."""
        if MODEL_PATH.exists():
            model_bundle = joblib.load(MODEL_PATH)
            self.model = model_bundle["model"]
            self.training_labels = np.array(model_bundle["labels"])
            self.class_names = list(model_bundle.get("class_names", self.class_names))
            self.training_sample_count = len(self.training_labels)
            return

        dataset_features, dataset_labels = self._load_training_image_features()
        if len(dataset_features) >= 3:
            self.model.n_neighbors = min(3, len(dataset_features))
            self.training_labels = np.array(dataset_labels)
            self.training_sample_count = len(self.training_labels)
            self.model.fit(np.array(dataset_features, dtype=np.float32), self.training_labels)
            return

        # Feature order:
        # mean_red, mean_green, mean_blue, green_ratio,
        # saturation_mean, value_mean, vertical_edge_strength
        training_features = np.array(
            [
                [0.44, 0.66, 0.40, 0.50, 0.42, 0.70, 0.58],
                [0.33, 0.55, 0.29, 0.57, 0.48, 0.63, 0.39],
                [0.72, 0.32, 0.31, 0.31, 0.66, 0.71, 0.28],
                [0.30, 0.52, 0.24, 0.60, 0.44, 0.52, 0.82],
                [0.36, 0.62, 0.28, 0.61, 0.51, 0.68, 0.35],
                [0.38, 0.62, 0.42, 0.54, 0.34, 0.70, 0.38],
                [0.41, 0.68, 0.36, 0.55, 0.40, 0.72, 0.42],
                [0.35, 0.64, 0.30, 0.60, 0.46, 0.66, 0.47],
                [0.43, 0.60, 0.34, 0.54, 0.36, 0.62, 0.30],
                [0.70, 0.38, 0.34, 0.32, 0.62, 0.70, 0.35],
                [0.34, 0.62, 0.36, 0.57, 0.43, 0.66, 0.36],
                [0.46, 0.63, 0.42, 0.50, 0.30, 0.68, 0.31],
                [0.67, 0.36, 0.34, 0.33, 0.58, 0.72, 0.34],
                [0.39, 0.58, 0.35, 0.54, 0.38, 0.61, 0.52],
                [0.42, 0.64, 0.38, 0.55, 0.42, 0.69, 0.28],
            ],
            dtype=np.float32,
        )
        labels = np.array(self.class_names)
        self.training_labels = labels
        self.training_sample_count = len(labels)
        self.model.fit(training_features, labels)

    def _load_training_image_features(self) -> tuple[list[np.ndarray], list[str]]:
        """Load optional real training images from training_images/<plant name>/."""
        features = []
        labels = []

        if not TRAINING_IMAGE_DIR.exists():
            return features, labels

        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for plant_name in self.class_names:
            plant_dir = TRAINING_IMAGE_DIR / plant_name
            if not plant_dir.is_dir():
                continue

            for image_path in plant_dir.iterdir():
                if image_path.suffix.lower() not in allowed_extensions:
                    continue
                try:
                    normalized_image = self.preprocess_image(str(image_path))
                    features.append(self.extract_features(normalized_image))
                    labels.append(plant_name)
                except (FileNotFoundError, ValueError):
                    continue

        return features, labels

    def preprocess_image(self, image_path: str, target_size: tuple[int, int] = (128, 128)) -> np.ndarray:
        """Resize image and normalize pixel values as required by the architecture."""
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        try:
            with Image.open(image_file) as pil_image:
                rgb_image = pil_image.convert("RGB")
        except Exception as error:
            raise ValueError("The file is not a valid image or cannot be opened.") from error

        image_array = np.array(rgb_image)
        resized = cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def extract_features(self, normalized_image: np.ndarray) -> np.ndarray:
        """Convert the normalized image into a plant image feature vector."""
        if normalized_image.ndim != 3 or normalized_image.shape[2] != 3:
            raise ValueError("Expected an RGB image with 3 color channels.")

        mean_rgb = normalized_image.mean(axis=(0, 1))
        std_rgb = normalized_image.std(axis=(0, 1))
        total_intensity = float(mean_rgb.sum()) + 1e-8
        green_ratio = float(mean_rgb[1] / total_intensity)

        rgb_uint8 = (normalized_image * 255).astype(np.uint8)
        hsv_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
        hsv_image = hsv_uint8.astype(np.float32) / 255.0
        mean_hsv = hsv_image.mean(axis=(0, 1))
        std_hsv = hsv_image.std(axis=(0, 1))

        hue_hist = cv2.calcHist([hsv_uint8], [0], None, [18], [0, 180]).flatten()
        saturation_hist = cv2.calcHist([hsv_uint8], [1], None, [8], [0, 256]).flatten()
        value_hist = cv2.calcHist([hsv_uint8], [2], None, [8], [0, 256]).flatten()
        hue_hist = hue_hist / (hue_hist.sum() + 1e-8)
        saturation_hist = saturation_hist / (saturation_hist.sum() + 1e-8)
        value_hist = value_hist / (value_hist.sum() + 1e-8)

        gray_image = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        sobel_vertical = cv2.Sobel(gray_image, cv2.CV_32F, dx=1, dy=0, ksize=3)
        sobel_horizontal = cv2.Sobel(gray_image, cv2.CV_32F, dx=0, dy=1, ksize=3)
        vertical_edge_strength = np.mean(np.abs(sobel_vertical)) / 255.0
        horizontal_edge_strength = np.mean(np.abs(sobel_horizontal)) / 255.0
        edge_magnitude = cv2.magnitude(sobel_vertical, sobel_horizontal)
        edge_mean = np.mean(edge_magnitude) / 255.0
        edge_std = np.std(edge_magnitude) / 255.0

        hue = hsv_uint8[:, :, 0]
        saturation = hsv_uint8[:, :, 1]
        value = hsv_uint8[:, :, 2]
        green_mask = (hue >= 35) & (hue <= 90) & (saturation > 35) & (value > 45)
        yellow_mask = (hue >= 18) & (hue < 35) & (saturation > 45) & (value > 70)
        red_mask = ((hue <= 10) | (hue >= 165)) & (saturation > 45) & (value > 50)
        dark_mask = value < 55
        light_mask = value > 190

        thumbnail = cv2.resize(rgb_uint8, (24, 24), interpolation=cv2.INTER_AREA)
        thumbnail_features = (thumbnail.astype(np.float32) / 255.0).flatten()

        feature_vector = np.concatenate(
            [
                mean_rgb,
                std_rgb,
                mean_hsv,
                std_hsv,
                np.array(
                    [
                        green_ratio,
                        float(green_mask.mean()),
                        float(yellow_mask.mean()),
                        float(red_mask.mean()),
                        float(dark_mask.mean()),
                        float(light_mask.mean()),
                        float(vertical_edge_strength),
                        float(horizontal_edge_strength),
                        float(edge_mean),
                        float(edge_std),
                    ],
                    dtype=np.float32,
                ),
                hue_hist.astype(np.float32),
                saturation_hist.astype(np.float32),
                value_hist.astype(np.float32),
                thumbnail_features.astype(np.float32),
            ]
        )
        return feature_vector.astype(np.float32)

    def predict(self, image_path: str) -> dict:
        """Predict the most likely plant name for an uploaded image."""
        normalized_image = self.preprocess_image(image_path)
        feature_vector = self.extract_features(normalized_image).reshape(1, -1)

        alternatives = []
        predicted_label = self.model.predict(feature_vector)[0]

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature_vector)[0]
            class_names = [str(class_name) for class_name in self.model.classes_]
            ranked = sorted(zip(class_names, probabilities), key=lambda item: item[1], reverse=True)
            confidence = float(ranked[0][1])
            alternatives = [
                {"plant_name": label, "confidence": float(probability)}
                for label, probability in ranked[:3]
            ]
        else:
            neighbor_count = min(3, self.training_sample_count)
            distances, indices = self.model.kneighbors(feature_vector, n_neighbors=neighbor_count)
            distance = float(distances[0][0])
            confidence = max(0.0, min(1.0, 1.0 - distance))
            seen_labels = set()

            for neighbor_distance, neighbor_index in zip(distances[0], indices[0]):
                label = str(self.training_labels[neighbor_index])
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                alternatives.append(
                    {
                        "plant_name": label,
                        "confidence": max(0.0, min(1.0, 1.0 - float(neighbor_distance))),
                    }
                )

        return {
            "plant_name": predicted_label,
            "confidence": confidence,
            "alternatives": alternatives,
        }

    def analyze_plant_health(self, image_path: str) -> dict:
        """Return a simple visual stress assessment for an uploaded plant image."""
        normalized_image = self.preprocess_image(image_path)
        rgb_image = (normalized_image * 255).astype(np.uint8)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        hue = hsv_image[:, :, 0]
        saturation = hsv_image[:, :, 1]
        value = hsv_image[:, :, 2]

        green_mask = (hue >= 35) & (hue <= 90) & (saturation > 35) & (value > 45)
        yellow_mask = (hue >= 18) & (hue < 35) & (saturation > 45) & (value > 70)
        brown_mask = (hue >= 5) & (hue < 25) & (saturation > 45) & (value < 150)
        dark_spot_mask = value < 45
        pale_mask = (saturation < 35) & (value > 130)

        total_pixels = float(rgb_image.shape[0] * rgb_image.shape[1])
        ratios = {
            "green": float(green_mask.sum() / total_pixels),
            "yellow": float(yellow_mask.sum() / total_pixels),
            "brown": float(brown_mask.sum() / total_pixels),
            "dark_spots": float(dark_spot_mask.sum() / total_pixels),
            "pale": float(pale_mask.sum() / total_pixels),
        }

        if ratios["brown"] > 0.12 or ratios["dark_spots"] > 0.18:
            issue = "Possible leaf spot, fungal stress, or dry/burnt tissue"
            care_hint = "Isolate the plant, remove badly damaged leaves, avoid wetting leaves, and improve airflow."
        elif ratios["yellow"] > 0.18:
            issue = "Possible yellowing from overwatering, nutrient deficiency, or low light"
            care_hint = "Check soil moisture, improve drainage, and provide bright indirect light."
        elif ratios["pale"] > 0.35 and ratios["green"] < 0.22:
            issue = "Possible pale growth from low light or nutrient deficiency"
            care_hint = "Move the plant to brighter indirect light and consider a mild balanced fertilizer."
        else:
            issue = "No strong visual disease pattern detected"
            care_hint = "Continue checking soil moisture, light, and leaf color regularly."

        return {
            "issue": issue,
            "care_hint": care_hint,
            "yellow_percent": ratios["yellow"] * 100,
            "brown_percent": ratios["brown"] * 100,
            "dark_spot_percent": ratios["dark_spots"] * 100,
            "note": "Heuristic image check only; confirm with close visual inspection.",
        }

"""Image preprocessing and lightweight plant image classification module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier


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
        self.model = KNeighborsClassifier(n_neighbors=1)
        self._train_model()

    def _train_model(self) -> None:
        """Train a small KNN model using handcrafted plant feature prototypes."""
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
        self.model.fit(training_features, labels)

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
        """Convert the normalized image into a compact feature vector."""
        if normalized_image.ndim != 3 or normalized_image.shape[2] != 3:
            raise ValueError("Expected an RGB image with 3 color channels.")

        mean_rgb = normalized_image.mean(axis=(0, 1))
        total_intensity = float(mean_rgb.sum()) + 1e-8
        green_ratio = float(mean_rgb[1] / total_intensity)

        hsv_image = cv2.cvtColor((normalized_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_image = hsv_image.astype(np.float32) / 255.0
        saturation_mean = float(hsv_image[:, :, 1].mean())
        value_mean = float(hsv_image[:, :, 2].mean())

        gray_image = cv2.cvtColor((normalized_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sobel_vertical = cv2.Sobel(gray_image, cv2.CV_32F, dx=1, dy=0, ksize=3)
        vertical_edge_strength = float(np.mean(np.abs(sobel_vertical)) / 255.0)

        feature_vector = np.array(
            [
                float(mean_rgb[0]),
                float(mean_rgb[1]),
                float(mean_rgb[2]),
                green_ratio,
                saturation_mean,
                value_mean,
                vertical_edge_strength,
            ],
            dtype=np.float32,
        )
        return feature_vector

    def predict(self, image_path: str) -> dict:
        """Predict the most likely plant name for an uploaded image."""
        normalized_image = self.preprocess_image(image_path)
        feature_vector = self.extract_features(normalized_image).reshape(1, -1)

        predicted_label = self.model.predict(feature_vector)[0]
        distances, _ = self.model.kneighbors(feature_vector, n_neighbors=1)
        distance = float(distances[0][0])
        confidence = max(0.0, min(1.0, 1.0 - distance))

        return {
            "plant_name": predicted_label,
            "confidence": confidence,
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

"""Flask web frontend for the Plant Care Assistant project."""

from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, render_template, request

from database_utils import (
    get_available_plants,
    get_plant_care_info,
    load_plant_database,
    search_plant_database,
)
from image_classifier import PlantImageClassifier


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

DATABASE = load_plant_database()
CLASSIFIER = PlantImageClassifier()
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _allowed_file(filename: str) -> bool:
    """Validate image file extensions before classification."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the home page and handle both search flows."""
    result = None
    error_message = None
    active_tab = "text"
    confidence = None
    matched_by = None
    search_query = ""
    suggestions = []
    health_analysis = None

    if request.method == "POST":
        action = request.form.get("action", "text-search")

        if action == "text-search":
            active_tab = "text"
            plant_name = request.form.get("plant_name", "").strip()
            search_query = plant_name

            if not plant_name:
                error_message = "Please enter a plant name before searching."
            else:
                search_result = search_plant_database(DATABASE, plant_name)
                result = search_result["plant"]
                suggestions = search_result["suggestions"]
                matched_by = search_result["matched_by"]
                if result is None:
                    error_message = f"No plant care profile found for '{plant_name}'."

        elif action == "image-search":
            active_tab = "image"
            image_file = request.files.get("plant_image")

            if image_file is None or image_file.filename == "":
                error_message = "Please upload an image file."
            elif not _allowed_file(image_file.filename):
                error_message = "Unsupported file type. Upload JPG, JPEG, PNG, BMP, or WEBP."
            else:
                temp_path = None
                try:
                    suffix = Path(image_file.filename).suffix.lower() or ".png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        image_file.save(temp_file)
                        temp_path = temp_file.name

                    prediction = CLASSIFIER.predict(temp_path)
                    confidence = prediction["confidence"]
                    health_analysis = CLASSIFIER.analyze_plant_health(temp_path)
                    result = get_plant_care_info(DATABASE, prediction["plant_name"])

                    if result is None:
                        error_message = (
                            f"The model predicted '{prediction['plant_name']}', "
                            "but no matching care profile exists."
                        )
                except Exception as error:  # pragma: no cover - defensive fallback
                    error_message = f"Unable to classify the image. {error}"
                finally:
                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink()

    return render_template(
        "index.html",
        result=result,
        error_message=error_message,
        active_tab=active_tab,
        confidence=confidence,
        matched_by=matched_by,
        search_query=search_query,
        suggestions=suggestions,
        health_analysis=health_analysis,
        plants=get_available_plants(DATABASE),
        plant_count=len(DATABASE),
    )


if __name__ == "__main__":
    app.run(debug=True)

"""Command-line entry point for the Plant Care Assistant project."""

from database_utils import get_plant_care_info, load_plant_database
from image_classifier import PlantImageClassifier
from utils import (
    format_plant_details,
    print_header,
    prompt_image_path,
    prompt_menu_choice,
    prompt_plant_name,
)


def handle_text_search(database: dict) -> None:
    """Search the knowledge base using a plant name entered by the user."""
    plant_name = prompt_plant_name()
    plant_info = get_plant_care_info(database, plant_name)

    if plant_info is None:
        print(f"\nPlant '{plant_name}' was not found in the knowledge database.\n")
        return

    print(format_plant_details(plant_info["plant_name"], plant_info))


def handle_image_upload(database: dict, classifier: PlantImageClassifier) -> None:
    """Classify an uploaded image and display the matching plant care profile."""
    image_path = prompt_image_path()

    try:
        prediction = classifier.predict(image_path)
        health_analysis = classifier.analyze_plant_health(image_path)
    except FileNotFoundError:
        print("\nThe image file could not be found. Please check the path and try again.\n")
        return
    except ValueError as error:
        print(f"\nUnable to process image: {error}\n")
        return
    except Exception as error:  # pragma: no cover - defensive fallback
        print(f"\nUnexpected error while classifying image: {error}\n")
        return

    plant_name = prediction["plant_name"]
    plant_info = get_plant_care_info(database, plant_name)

    if plant_info is None:
        print(f"\nPlant '{plant_name}' was predicted, but no care profile exists in the database.\n")
        return

    print(f"\nModel Confidence: {prediction['confidence']:.2%}")
    print(f"Image Health Check: {health_analysis['issue']}")
    print(f"Care Hint: {health_analysis['care_hint']}")
    print(format_plant_details(plant_info["plant_name"], plant_info))


def main() -> None:
    """Run the main CLI loop for the project."""
    print_header()
    database = load_plant_database()
    classifier = PlantImageClassifier()

    while True:
        print("1. Text Search Plant")
        print("2. Upload Plant Image")
        print("3. Exit")

        choice = prompt_menu_choice()

        if choice == "1":
            handle_text_search(database)
        elif choice == "2":
            handle_image_upload(database, classifier)
        elif choice == "3":
            print("\nThank you for using Plant Care Assistant.")
            break


if __name__ == "__main__":
    main()

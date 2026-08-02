"""Reusable CLI helpers for user input and display formatting."""


def print_header() -> None:
    """Display a friendly application header."""
    print("\n" + "=" * 50)
    print("         Plant Care Assistant")
    print("=" * 50 + "\n")


def prompt_menu_choice() -> str:
    """Ask the user for a valid menu option."""
    valid_choices = {"1", "2", "3"}

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in valid_choices:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def prompt_plant_name() -> str:
    """Get the plant name from the user."""
    while True:
        plant_name = input("\nEnter plant name: ").strip()
        if plant_name:
            return plant_name
        print("Plant name cannot be empty. Please try again.")


def prompt_image_path() -> str:
    """Get the image file path from the user."""
    while True:
        image_path = input("\nEnter image path: ").strip().strip('"').strip("'")
        if image_path:
            return image_path
        print("Image path cannot be empty. Please try again.")


def format_plant_details(plant_name: str, plant_info: dict) -> str:
    """Return the plant care details in the required output format."""
    return (
        f"\nPlant Identified: {plant_name}\n\n"
        f"Scientific Name: {plant_info.get('scientific_name', 'Not specified')}\n"
        f"Family: {plant_info.get('family', 'Not specified')}\n"
        f"Water Frequency: {plant_info['water_frequency']}\n"
        f"Sunlight: {plant_info['sunlight_requirement']}\n"
        f"Indoor or Outdoor: {plant_info.get('location', 'Not specified')}\n"
        f"Soil Type: {plant_info.get('soil_type', 'Not specified')}\n"
        f"Temperature: {plant_info.get('temperature', 'Not specified')}\n"
        f"Maintenance: {plant_info['maintenance_tips']}\n"
        f"Pro Tip: {plant_info['unique_pro_tip']}\n"
        f"Difficulty: {plant_info.get('difficulty', 'Not specified')}\n"
        f"Common Problem: {plant_info.get('common_problem', 'Not specified')}\n"
        f"Disease or Deficiency: {plant_info.get('disease_or_deficiency', 'Not specified')}\n"
    )

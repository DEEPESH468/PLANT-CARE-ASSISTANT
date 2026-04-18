# Plant Care Assistant

Plant Care Assistant is a modular Python project that helps users identify plants and view plant care instructions using either:

- Text search
- Image upload with preprocessing and classification

The project now includes:

- A Flask-based web frontend for demos and presentation
- The original CLI version for terminal-based usage
- Fuzzy text search with clickable suggestions for close plant names
- Expanded plant care profiles with difficulty and common-problem notes
- Scientific names, botanical family, air-purifying value, soil, temperature, and plant-part details
- Indoor or outdoor suitability for each plant
- A simple image-based plant health check for visible yellowing, browning, dark spots, or pale growth

The project follows the system architecture shown in the flowchart:

1. User opens the application
2. User chooses an input method
3. System either searches the plant care knowledge base or classifies an uploaded image
4. The matching plant care profile is displayed, or close suggestions are shown

## Project Structure

```text
plant_care_assistant/
├── main.py
├── app.py
├── image_classifier.py
├── database_utils.py
├── plants_database.json
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── utils.py
├── requirements.txt
└── README.md
```

## How Each Module Works

### `app.py`
- Runs the Flask web server
- Handles both text search and image upload form submissions
- Renders the frontend and displays plant care results in the browser

### `main.py`
- Runs the command-line menu
- Handles the text search flow
- Handles the image upload flow
- Connects the classifier and database search modules

### `image_classifier.py`
- Loads and preprocesses uploaded images
- Resizes the image
- Normalizes pixel values
- Extracts image features
- Uses a simple `scikit-learn` KNN classifier to predict the plant name

### `database_utils.py`
- Loads plant care data from the JSON database
- Searches the plant knowledge base using case-insensitive matching
- Supports fuzzy matching for misspelled plant names
- Returns the full plant care profile

### `utils.py`
- Provides helper functions for input validation
- Formats the final plant care output
- Prints the application header and menu prompts

### `plants_database.json`
- Stores plant care information for:
  - Aloe Vera
  - Tulsi
  - Rose
  - Snake Plant
  - Money Plant
  - Peace Lily
  - Spider Plant
  - Areca Palm
  - Jade Plant
  - Hibiscus
  - Syngonium
  - Aglaonema
  - Anthurium
  - Satavar
  - Patharchatta

## Installation Commands

Open a terminal in VS Code and run:

```bash
cd plant_care_assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the virtual environment with:

```powershell
venv\Scripts\Activate.ps1
```

## How to Run the Web Frontend

From the `plant_care_assistant` folder:

```bash
python3 app.py
```

Then open this URL in your browser:

```text
http://127.0.0.1:5000
```

## How to Run the CLI Version

If you still want the terminal version:

```bash
python3 main.py
```

## Program Menu

When the CLI program starts, it shows:

```text
1. Text Search Plant
2. Upload Plant Image
3. Exit
```

## Example Output

```text
Plant Identified: Aloe Vera

Water Frequency: Water every 10-14 days and let the soil dry between watering
Sunlight: Bright indirect sunlight or mild morning sun
Indoor or Outdoor: Outdoor and indoor
Maintenance: Use fast-draining soil and a pot with drainage holes
Pro Tip: Rotate the pot every week so the leaves grow evenly toward light
Difficulty: Easy
Common Problem: Soft brown leaves usually mean the plant is being overwatered
```

## Steps to Run in VS Code

1. Open VS Code.
2. Open the project folder `pbl_pythonlab`.
3. Open the integrated terminal from `Terminal > New Terminal`.
4. Run:

```bash
cd plant_care_assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

5. Open `http://127.0.0.1:5000` in your browser.
6. Use the text search form or image upload form on the page.

## How to Add More Plants

To add more plants:

1. Open `plants_database.json`
2. Add a new plant entry in the same JSON format
3. Make sure each plant includes:
   - `water_frequency`
   - `sunlight_requirement`
   - `maintenance_tips`
   - `unique_pro_tip`
   - `difficulty`
   - `common_problem`
   - `scientific_name`
   - `family`
   - `soil_type`
   - `temperature`
   - `location`
   - `disease_or_deficiency`

Example:

```json
"Lavender": {
  "aliases": ["Lavandula"],
  "scientific_name": "Lavandula angustifolia",
  "family": "Lamiaceae",
  "water_frequency": "Water once or twice a week",
  "sunlight_requirement": "Full sunlight",
  "location": "Outdoor",
  "maintenance_tips": "Use sandy, well-drained soil",
  "unique_pro_tip": "Do not overwater because lavender prefers dry conditions",
  "difficulty": "Moderate",
  "common_problem": "Root rot can happen quickly if the soil stays wet",
  "soil_type": "Sandy, well-drained soil",
  "temperature": "18 deg C to 30 deg C",
  "disease_or_deficiency": "Root rot, fungal leaf spot, and weak growth from low sunlight"
}
```

## Notes

- The image classifier is intentionally lightweight for a university project and works without a separate training dataset file.
- You can improve recognition accuracy later by training the classifier on real plant images.
- The text search flow supports exact, case-insensitive, and fuzzy matching.
- The image health check is heuristic and should be treated as a first clue, not a trained disease diagnosis model.
- The web frontend reuses the same backend modules, so your architecture stays modular and easy to explain in a viva or report.

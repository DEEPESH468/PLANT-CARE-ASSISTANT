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
├── train_image_model.py
├── scripts/
│   ├── import_huggingface_plants.py
│   └── prepare_dataset.py
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
- Loads a trained model from `plant_image_model.joblib` when available
- Falls back to real images in `training_images/` or built-in demo prototypes
- Uses a `scikit-learn` KNN classifier to predict the plant name

### `train_image_model.py`
- Trains the image classifier from local folders
- Saves the trained model as `plant_image_model.joblib`
- Prints image counts and validation accuracy when enough data is available

### `scripts/prepare_dataset.py`
- Copies matching plant images from an extracted dataset into `training_images/`
- Supports class-name matching for plants such as Aloe Vera, Snake Plant, Money Plant, Peace Lily, Syngonium, Anthurium, and others
- Keeps downloaded datasets separate from source code

### `scripts/import_huggingface_plants.py`
- Imports selected classes from `funkepal/medicinal_plant_images`
- Adds extra training images for Aloe Vera, Hibiscus, Rose, and Tulsi
- Helps cover app plants that are missing from the houseplant species dataset

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

## How to Train Image Recognition with a Dataset

The app can use a locally trained model for better image upload accuracy.

Recommended dataset choice:

- `kakasher/house-plant-species` on Hugging Face is better for household plant species recognition.
- `hadyahmed00/plants-leafs-dataset` on Kaggle is better for leaf disease or healthy/unhealthy detection because it focuses on plant leaves and disease classes.

Keep downloaded datasets and trained model files out of GitHub. The repo ignores:

```text
training_images/
datasets/
plant_image_model.joblib
```

### 1. Download and extract a dataset

Create a local dataset folder:

```bash
mkdir -p datasets
```

For the Hugging Face houseplant dataset, download and extract it into:

```text
plant_care_assistant/datasets/house-plant-species/
```

One way to download it from the terminal is:

```bash
mkdir -p datasets/house-plant-species
curl -L "https://huggingface.co/datasets/kakasher/house-plant-species/resolve/main/house_plant_species.tar" -o datasets/house_plant_species.tar
tar -xf datasets/house_plant_species.tar -C datasets/house-plant-species
```

For the Kaggle plant leaves dataset, download and extract it into:

```text
plant_care_assistant/datasets/plants-leafs-dataset/
```

From Kaggle, download the ZIP from:

```text
https://www.kaggle.com/datasets/hadyahmed00/plants-leafs-dataset
```

Then move/extract it into:

```text
plant_care_assistant/datasets/plants-leafs-dataset/
```

### 2. Prepare matching training folders

From the `plant_care_assistant` folder, run one of these:

```bash
python3 scripts/prepare_dataset.py datasets/house-plant-species --max-images-per-class 80
```

or:

```bash
python3 scripts/prepare_dataset.py datasets/plants-leafs-dataset --max-images-per-class 80
```

To add extra images for Aloe Vera, Hibiscus, Rose, and Tulsi from Hugging Face:

```bash
python3 scripts/import_huggingface_plants.py --max-images-per-class 160
```

This creates folders like:

```text
training_images/
├── Aloe Vera/
├── Money Plant/
├── Peace Lily/
├── Snake Plant/
└── Syngonium/
```

### 3. Train the model

```bash
python3 train_image_model.py
```

This creates:

```text
plant_image_model.joblib
```

### 4. Run the app again

Restart Flask after training:

```bash
python3 app.py
```

The upload feature will automatically use `plant_image_model.joblib`.

If the dataset does not contain some plants from the app database, those plants cannot be recognized accurately from photos until you add labeled images for them.

Current trained dataset coverage:

```text
Aglaonema
Aloe Vera
Anthurium
Areca Palm
Hibiscus
Jade Plant
Money Plant
Patharchatta
Peace Lily
Rose
Snake Plant
Tulsi
```

Still missing from the image dataset:

```text
Satavar
Spider Plant
Syngonium
```

## SDG Alignment

This project is mainly aligned with **Sustainable Development Goal 15: Life on Land**.

The Plant Care Assistant supports plant health by helping users understand watering needs, sunlight requirements, soil conditions, common diseases, and mineral deficiencies. By making plant care information easier to access, the project encourages better care of household plants, garden plants, and small-scale green spaces.

The project also supports these related SDGs:

- **SDG 2: Zero Hunger** - healthy plant-care guidance can support small gardens, kitchen gardening, and better plant productivity.
- **SDG 12: Responsible Consumption and Production** - proper plant maintenance can reduce plant waste caused by overwatering, poor sunlight, disease, or neglect.
- **SDG 13: Climate Action** - encouraging plant care and greenery builds awareness of environment-friendly habits.

For project presentation, the best primary SDG to mention is:

```text
SDG 15: Life on Land
```

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

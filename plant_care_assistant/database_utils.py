"""Utility helpers for loading and querying the plant care knowledge database."""

from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import pandas as pd


DATABASE_PATH = Path(__file__).resolve().parent / "plants_database.json"


def load_plant_database() -> dict:
    """Load the JSON database into memory."""
    try:
        with DATABASE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Database file not found at {DATABASE_PATH}") from error
    except json.JSONDecodeError as error:
        raise ValueError("The plant database file contains invalid JSON.") from error

    return data


def _normalize_name(plant_name: str) -> str:
    """Normalize user input for case-insensitive matching."""
    return " ".join(plant_name.strip().lower().split())


def get_plant_care_info(database: dict, plant_name: str) -> Optional[dict]:
    """Return plant care details for a given plant name if it exists."""
    search_result = search_plant_database(database, plant_name)
    return search_result["plant"]


def search_plant_database(database: dict, plant_name: str) -> dict:
    """Search by exact or fuzzy plant name and include suggestions if needed."""
    normalized_input = _normalize_name(plant_name)

    records = []
    for name, details in database.items():
        search_terms = [
            name,
            details.get("scientific_name", ""),
            *details.get("aliases", []),
        ]
        for term in search_terms:
            if term:
                records.append(
                    {
                        "plant_name": name,
                        **details,
                        "normalized_name": _normalize_name(term),
                        "search_label": term,
                    }
                )

    dataframe = pd.DataFrame(records)

    match = dataframe[dataframe["normalized_name"] == normalized_input]
    if not match.empty:
        result = match.iloc[0].drop(labels=["normalized_name", "search_label"]).to_dict()
        return {"plant": result, "suggestions": [], "matched_by": "exact"}

    normalized_names = dataframe["normalized_name"].tolist()
    close_matches = get_close_matches(normalized_input, normalized_names, n=3, cutoff=0.75)
    if close_matches:
        closest = close_matches[0]
        fuzzy_match = dataframe[dataframe["normalized_name"] == closest].iloc[0]
        result = fuzzy_match.drop(labels=["normalized_name", "search_label"]).to_dict()
        suggestions = []
        for match_name in close_matches:
            suggestion = dataframe[dataframe["normalized_name"] == match_name].iloc[0]["plant_name"]
            if suggestion not in suggestions:
                suggestions.append(suggestion)
        return {"plant": result, "suggestions": suggestions, "matched_by": "fuzzy"}

    suggestions = get_close_matches(
        plant_name,
        get_available_plants(database),
        n=3,
        cutoff=0.35,
    )
    return {"plant": None, "suggestions": suggestions, "matched_by": None}


def get_available_plants(database: dict) -> list[str]:
    """Return all available plant names sorted alphabetically."""
    return sorted(database.keys())

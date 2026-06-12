"""
Model Registry
---------------
After training, the best model gets registered here with:
  - Version number
  - Performance metrics
  - Feature list it was trained on
  - Status: challenger → champion (after review)
We save the model as a .pkl file + a JSON metadata card.
"""

import json
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger


PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_DIR = PROJECT_ROOT / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_INDEX = REGISTRY_DIR / "registry_index.json"


def _load_index() -> list:
    if REGISTRY_INDEX.exists():
        return json.loads(REGISTRY_INDEX.read_text())
    return []


def _save_index(index: list):
    REGISTRY_INDEX.write_text(json.dumps(index, indent=2))


def register_model(
    model,
    model_name: str,
    metrics: dict,
    features: list,
    params: dict = None,
    notes: str = "",
) -> dict:
    """
    Registers a trained model to the local registry.
    Assigns a version number, saves model + metadata card.
    """
    index = _load_index()
    version = len([m for m in index if m["model_name"] == model_name]) + 1

    model_id = f"{model_name}_v{version}"
    model_path = REGISTRY_DIR / f"{model_id}.pkl"
    meta_path  = REGISTRY_DIR / f"{model_id}_card.json"

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Build model card (standard in industry — documents what the model is)
    model_card = {
        "model_id":    model_id,
        "model_name":  model_name,
        "version":     version,
        "status":      "challenger",      # starts as challenger, promoted to champion after review
       "registered": datetime.now(timezone.utc).isoformat(),
        "metrics":     metrics,
        "features":    features,
        "n_features":  len(features),
        "params":      params or {},
        "notes":       notes,
        "model_path":  str(model_path),
    }
    meta_path.write_text(json.dumps(model_card, indent=2))

    # Update index
    index.append(model_card)
    _save_index(index)

    logger.success(f"Registered: {model_id} | AUC: {metrics.get('auc')} | "
                   f"Status: challenger")
    return model_card


def promote_to_champion(model_id: str):
    """
    Promotes a challenger model to champion (production).
    Demotes the previous champion to archived.
    In production: this triggers a deployment pipeline.
    """
    index = _load_index()

    for entry in index:
        if entry["status"] == "champion":
            entry["status"] = "archived"
            logger.info(f"Archived previous champion: {entry['model_id']}")

    for entry in index:
        if entry["model_id"] == model_id:
            entry["status"] = "champion"
            meta_path = REGISTRY_DIR / f"{model_id}_card.json"
            meta_path.write_text(json.dumps(entry, indent=2))
            logger.success(f"Promoted to champion: {model_id}")

    _save_index(index)


def load_champion(model_name: str):
    """Loads the current champion model for serving."""
    index = _load_index()
    champions = [m for m in index
                 if m["model_name"] == model_name and m["status"] == "champion"]
    if not champions:
        raise ValueError(f"No champion model found for: {model_name}")

    champion = champions[-1]
    with open(champion["model_path"], "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loaded champion: {champion['model_id']} | "
                f"AUC: {champion['metrics'].get('auc')}")
    return model, champion


def list_models():
    index = _load_index()
    if not index:
        logger.warning("No models registered yet.")
        return

    logger.info("\n── Model Registry ──")
    for m in index:
        logger.info(f"  {m['model_id']:30s} | "
                    f"AUC: {m['metrics'].get('auc', 'N/A')} | "
                    f"Status: {m['status']:12s} | "
                    f"Registered: {m['registered'][:10]}")


if __name__ == "__main__":
    list_models()
"""Portable, inference-only reader for the user's custom shared-tree MT-GBM.

Each booster is a ONE-tree LightGBM shared topology. The user's fitted class
owns task-specific Newton leaf values, which ordinary LightGBM probabilities
DO NOT contain. It would be incorrect to use booster.predict() as risk output.
"""
from __future__ import annotations

import gzip
import json
from hashlib import sha256
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.special import expit

FORMAT = "tips_shared_tree_mtgbm_v1"
EXPECTED_FEATURES = ["Age", "PLT", "Hb", "Alb", "TBIL", "Scr", "INR", "PT_prolongation"]
EXPECTED_OUTCOMES = ["Rebleeding_365", "OHE_365", "Mortality_365"]


class PortableMTGBM:
    def __init__(self, bundle_path: str | Path):
        path = Path(bundle_path)
        self.fingerprint = sha256(path.read_bytes()).hexdigest()[:16]
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != FORMAT:
            raise ValueError("Model-bundle format is not compatible with this app")
        if payload.get("features") != EXPECTED_FEATURES:
            raise ValueError("Model feature order differs from original Notebook")
        if payload.get("outcomes") != EXPECTED_OUTCOMES:
            raise ValueError("Model outcome order differs from original Notebook")
        self.features = payload["features"]
        self.outcomes = payload["outcomes"]
        self.reference = payload.get("training_reference", {})
        self.cutoffs = payload.get("oof_youden_cutoffs", {})
        self.training_versions = payload.get("training_versions", {})
        self.model_version = payload.get("model_version", "not specified")
        self.units_confirmed = bool(payload.get("units_confirmed", False))
        self.learning_rate = float(payload["learning_rate"])
        self.base_logits = np.asarray(payload["base_logits"], dtype=np.float64)
        if self.base_logits.shape != (len(self.outcomes),):
            raise ValueError("Invalid task intercept dimensions")
        self.trees = []
        self.leaf_values = []
        for idx, item in enumerate(payload["trees"]):
            tree = lgb.Booster(model_str=item["model"])
            values = np.asarray(item["leaf_values"], dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != len(self.outcomes):
                raise ValueError(f"Bad leaf-value matrix in tree {idx}")
            if tree.num_trees() != 1:
                raise ValueError(f"Expected one tree per boosting iteration; tree {idx}")
            self.trees.append(tree)
            self.leaf_values.append(values)
        if not self.trees:
            raise ValueError("Empty model bundle")
        importance = np.asarray(payload.get("shared_feature_importance", []), dtype=float)
        self.shared_feature_importance = (
            importance if importance.shape == (len(self.features),) else None
        )

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError("Expected an eight-column feature matrix in exact trained order")
        if not np.isfinite(X).all():
            raise ValueError("All eight input values must be finite numbers")
        logits = np.tile(self.base_logits, (X.shape[0], 1))
        for tree, values in zip(self.trees, self.leaf_values):
            leaf_ids = np.asarray(tree.predict(X, pred_leaf=True, num_threads=1)).reshape(-1)
            leaf_ids = leaf_ids.astype(np.int64)
            if ((leaf_ids < 0) | (leaf_ids >= values.shape[0])).any():
                raise RuntimeError("LightGBM leaf indexes mismatch exported task leaf values")
            logits += self.learning_rate * values[leaf_ids]
        return expit(logits)

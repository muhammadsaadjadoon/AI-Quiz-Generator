"""
difficulty_model.py
────────────────────────────────────────────────────────────────────
Custom PyTorch neural network for MCQ difficulty classification
(easy / medium / hard).

Design:
    Sentence-Transformer embedding (384-dim, frozen, pretrained)
        -> Linear(384, 128) -> ReLU -> Dropout
        -> Linear(128, 64)  -> ReLU -> Dropout
        -> Linear(64, 3)    -> logits (easy/medium/hard)

This is a genuine small feed-forward neural network that YOU can
fine-tune on your own labeled data using train_difficulty.py.
The sentence embedding model itself stays frozen (we don't fine-tune
the transformer, only the small classifier head) so training is fast
enough to run on a CPU-only laptop in a few minutes, even on a few
thousand rows.

If no fine-tuned weights are found on disk, this falls back to a
rule-based heuristic so the app still works out of the box before
you've trained anything.
────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("DIFFICULTY_MODEL")

LABELS = ["easy", "medium", "hard"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}

EMBED_DIM = 384  # matches all-MiniLM-L6-v2 sentence-transformers output
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "difficulty_weights.pt")


class DifficultyNet(nn.Module):
    """Small feed-forward classifier head on top of frozen sentence embeddings."""

    def __init__(self, embed_dim: int = EMBED_DIM, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DifficultyClassifier:
    """
    Wraps DifficultyNet + a sentence-embedding encoder so callers can
    just pass raw question text and get a difficulty label back.

    Falls back to a transparent heuristic (sentence length / vocabulary
    complexity) if no trained weights are present yet, so the rest of
    the app keeps working before any training has happened.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder
        self.model: Optional[DifficultyNet] = None
        self.device = torch.device("cpu")
        self._load_if_available()

    def _load_if_available(self):
        if os.path.exists(WEIGHTS_PATH):
            try:
                net = DifficultyNet()
                state = torch.load(WEIGHTS_PATH, map_location=self.device)
                net.load_state_dict(state)
                net.eval()
                self.model = net
                logger.info(f"✓ Loaded fine-tuned difficulty model from {WEIGHTS_PATH}")
            except Exception as e:
                logger.warning(f"Could not load difficulty weights ({e}); using heuristic fallback")
                self.model = None
        else:
            logger.info(
                "ℹ No fine-tuned difficulty_weights.pt found yet — using heuristic fallback. "
                "Run train_difficulty.py with your own labeled data to enable the neural classifier."
            )

    def _heuristic(self, text: str) -> str:
        """Transparent rule-based fallback, used only until a model is trained."""
        words = text.split()
        n = len(words)
        long_words = sum(1 for w in words if len(w) > 7)
        ratio = long_words / max(n, 1)
        if n <= 10 and ratio < 0.15:
            return "easy"
        if n <= 22 and ratio < 0.35:
            return "medium"
        return "hard"

    @torch.no_grad()
    def predict(self, text: str) -> str:
        if self.model is None or self.embedder is None:
            return self._heuristic(text)
        try:
            emb = self.embedder.encode(text)
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            logits = self.model(x)
            idx = int(torch.argmax(logits, dim=-1).item())
            return IDX2LABEL[idx]
        except Exception as e:
            logger.warning(f"Difficulty model inference failed ({e}); using heuristic")
            return self._heuristic(text)

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[str]:
        if self.model is None or self.embedder is None or not texts:
            return [self._heuristic(t) for t in texts]
        try:
            embs = self.embedder.encode(texts)
            x = torch.tensor(embs, dtype=torch.float32)
            logits = self.model(x)
            idxs = torch.argmax(logits, dim=-1).tolist()
            return [IDX2LABEL[i] for i in idxs]
        except Exception as e:
            logger.warning(f"Batch difficulty inference failed ({e}); using heuristic")
            return [self._heuristic(t) for t in texts]

    def is_trained(self) -> bool:
        return self.model is not None


def save_training_meta(history: dict, path: str = None):
    """Persist a small JSON sidecar with training run stats, used by /api/v1/ml-info."""
    path = path or os.path.join(os.path.dirname(__file__), "difficulty_train_meta.json")
    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write training meta: {e}")


def load_training_meta(path: str = None) -> Optional[dict]:
    path = path or os.path.join(os.path.dirname(__file__), "difficulty_train_meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

"""
train_difficulty.py
────────────────────────────────────────────────────────────────────
Trains / fine-tunes the DifficultyNet classifier on YOUR OWN labeled
data. This is the script that does the actual "deep learning fine-
tuning" part of the project: a real PyTorch training loop, with a
train/validation split, loss curves, and a saved checkpoint that
difficulty_model.py will automatically pick up afterwards.

────────────────────────────────────────────────────────────────────
HOW TO PROVIDE YOUR OWN DATA
────────────────────────────────────────────────────────────────────
Supported formats — give either of these via --data:

1) CSV file with two columns: text,difficulty
   Example (training_data.csv):

       text,difficulty
       "What is the capital of France?",easy
       "Explain how photosynthesis converts light energy into ATP.",medium
       "Critically evaluate the trade-offs between bias and variance in regularized regression.",hard

2) JSON file: a list of {"text": ..., "difficulty": ...} objects

       [
         {"text": "What is gravity?", "difficulty": "easy"},
         {"text": "Describe how neural networks backpropagate error.", "difficulty": "medium"}
       ]

difficulty must be one of: easy, medium, hard (case-insensitive).

────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────
    pip install -r requirements.txt
    python train_difficulty.py --data training_data.csv --epochs 25

Optional flags:
    --epochs        number of training epochs (default 25)
    --lr            learning rate (default 1e-3)
    --batch-size    batch size (default 16)
    --val-split     fraction held out for validation (default 0.15)
    --seed          random seed (default 42)

After training completes, difficulty_weights.pt is written next to
this script. The FastAPI app (app.py) and difficulty_model.py will
automatically load it on next startup — no other code changes needed.
────────────────────────────────────────────────────────────────────
"""

import os
import csv
import json
import argparse
import logging
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from difficulty_model import DifficultyNet, LABEL2IDX, LABELS, EMBED_DIM, WEIGHTS_PATH, save_training_meta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TRAIN_DIFFICULTY")


def load_dataset_file(path: str) -> List[Tuple[str, str]]:
    """Reads a CSV or JSON file of {text, difficulty} pairs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    rows: List[Tuple[str, str]] = []
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in reader.fieldnames or "difficulty" not in reader.fieldnames:
                raise ValueError("CSV must have 'text' and 'difficulty' columns")
            for row in reader:
                text = (row.get("text") or "").strip()
                diff = (row.get("difficulty") or "").strip().lower()
                if text and diff in LABEL2IDX:
                    rows.append((text, diff))
                elif text and diff:
                    logger.warning(f"Skipping row with unknown difficulty '{diff}': {text[:50]}")

    elif ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of {text, difficulty} objects")
        for item in data:
            text = str(item.get("text", "")).strip()
            diff = str(item.get("difficulty", "")).strip().lower()
            if text and diff in LABEL2IDX:
                rows.append((text, diff))
            elif text and diff:
                logger.warning(f"Skipping row with unknown difficulty '{diff}': {text[:50]}")
    else:
        raise ValueError("Data file must be .csv or .json")

    if not rows:
        raise ValueError("No valid rows found. Check your file format and difficulty labels.")

    return rows


class DifficultyDataset(Dataset):
    """Pre-computes sentence embeddings once, so training epochs are fast on CPU."""

    def __init__(self, rows: List[Tuple[str, str]], embedder):
        texts = [r[0] for r in rows]
        labels = [LABEL2IDX[r[1]] for r in rows]
        logger.info(f"Encoding {len(texts)} examples with sentence-transformer (one-time cost)…")
        embeddings = embedder.encode(texts, show_progress_bar=True)
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def class_balance_report(rows: List[Tuple[str, str]]):
    counts = {l: 0 for l in LABELS}
    for _, d in rows:
        counts[d] += 1
    logger.info(f"Class balance: {counts}")
    return counts


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import here (not top-level) so this script gives a clear error message
    # if sentence-transformers isn't installed yet, instead of crashing on import.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )

    logger.info("Loading pretrained sentence embedder (all-MiniLM-L6-v2)…")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info(f"Loading training data from {args.data}")
    rows = load_dataset_file(args.data)
    logger.info(f"Loaded {len(rows)} labeled examples")
    counts = class_balance_report(rows)

    for label, c in counts.items():
        if c == 0:
            logger.warning(
                f"⚠ No examples for class '{label}'. The model will never predict this "
                f"class well. Add some '{label}' examples to your data file for best results."
            )

    dataset = DifficultyDataset(rows, embedder)

    val_size = max(1, int(len(dataset) * args.val_split)) if len(dataset) >= 10 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        train_ds, val_ds = dataset, None
        logger.warning("Dataset is small (<10 rows) — skipping validation split")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

    model = DifficultyNet(embed_dim=EMBED_DIM, num_classes=len(LABELS))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None

    logger.info(f"Starting training: {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        val_loss, val_acc = None, None
        if val_loader:
            model.eval()
            v_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    v_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_loss = v_loss / max(len(val_loader), 1)
            val_acc = correct / max(total, 1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        log_line = f"Epoch {epoch:3d}/{args.epochs} | train_loss: {train_loss:.4f}"
        if val_loss is not None:
            log_line += f" | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.1f}%"
        logger.info(log_line)

    logger.info("=" * 60)

    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(final_state, WEIGHTS_PATH)
    logger.info(f"✓ Saved trained weights to {WEIGHTS_PATH}")

    save_training_meta({
        "examples": len(rows),
        "class_balance": counts,
        "epochs": args.epochs,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "best_val_loss": best_val_loss if best_state is not None else None,
    })
    logger.info("✓ Training complete. Restart app.py to load the fine-tuned model.")


def build_starter_dataset(path: str):
    """Writes a small example training_data.csv so users have something to start from/extend."""
    rows = [
        ("What is the capital of France?", "easy"),
        ("Who wrote Romeo and Juliet?", "easy"),
        ("What is the chemical symbol for water?", "easy"),
        ("Define the term photosynthesis.", "easy"),
        ("What color is chlorophyll?", "easy"),
        ("Explain how enzymes lower the activation energy of a reaction.", "medium"),
        ("Describe the difference between mitosis and meiosis.", "medium"),
        ("How does supervised learning differ from unsupervised learning?", "medium"),
        ("What factors influence the rate of diffusion across a cell membrane?", "medium"),
        ("Explain the role of mitochondria in cellular respiration.", "medium"),
        ("Critically evaluate the assumptions underlying the efficient market hypothesis.", "hard"),
        ("Derive the relationship between entropy and information content in a noisy channel.", "hard"),
        ("Analyze the trade-offs between bias and variance in regularized regression models.", "hard"),
        ("Assess the long-term ecological consequences of large-scale deforestation on biodiversity.", "hard"),
        ("Synthesize evidence from cognitive neuroscience to explain the neural basis of working memory.", "hard"),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "difficulty"])
        writer.writerows(rows)
    logger.info(f"✓ Wrote starter dataset with {len(rows)} examples to {path}")
    logger.info("  Add your own rows (the more, the better) then re-run training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the MCQ difficulty classifier on your own data")
    parser.add_argument("--data", type=str, default=None, help="Path to your training_data.csv or .json file")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--make-starter-data", type=str, default=None,
        help="Instead of training, just write a small starter training_data.csv to this path"
    )
    args = parser.parse_args()

    if args.make_starter_data:
        build_starter_dataset(args.make_starter_data)
    elif args.data:
        train(args)
    else:
        parser.error("Provide --data path/to/your_data.csv (or --make-starter-data to generate an example first)")

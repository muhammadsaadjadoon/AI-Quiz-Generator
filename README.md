# UoH AI Quiz Engine — Deep Learning Edition

A deep-learning-powered MCQ generator: paste text OR upload a PDF, and it
generates multiple-choice questions using a real neural pipeline (not regex
templates) — pretrained transformers + a custom fine-tunable PyTorch network.

## Files

| File                   | Purpose                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| `app.py`               | FastAPI backend — auth, quiz generation (text/PDF), history, ML info   |
| `ml_engine.py`         | Deep learning core — sentence embeddings, T5 question generation, KMeans, distractor mining |
| `difficulty_model.py`  | Custom PyTorch neural network for difficulty classification (easy/medium/hard) |
| `train_difficulty.py`  | Standalone script to fine-tune the difficulty network on YOUR OWN labeled data |
| `requirements.txt`     | All dependencies |

## How the "deep learning" actually works

1. **Sentence-Transformers (`all-MiniLM-L6-v2`)** turns every sentence of
   your text into a 384-dimensional embedding. These embeddings power:
   - semantic importance ranking (which sentences matter most)
   - KMeans topic clustering
   - distractor mining (finding *related but wrong* answers, not random noise)

2. **T5 transformer (`valhalla/t5-small-qg-hl`, falls back to `t5-small`)**
   is a pretrained sequence-to-sequence model that generates a natural
   question for a highlighted answer span — this is genuine neural text
   generation, not a sentence-pattern template.

3. **`DifficultyNet`** (in `difficulty_model.py`) is a small feed-forward
   PyTorch network (384 → 128 → 64 → 3) sitting on top of the same sentence
   embeddings. Out of the box it falls back to a transparent length/vocabulary
   heuristic — **train it on your own data to get the real neural classifier**.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
python app.py
```

First run downloads the pretrained models from HuggingFace (a few hundred MB,
needs internet once). After that everything runs fully offline and on CPU.

API docs: http://127.0.0.1:8000/docs

## Fine-tuning the difficulty classifier on your own data

1. Get a starter file to see the expected format:
   ```bash
   python train_difficulty.py --make-starter-data training_data.csv
   ```
2. Open `training_data.csv` and add your own rows (more is better — aim for
   at least 100–200 examples spread across easy/medium/hard if you can).
   Format:
   ```csv
   text,difficulty
   "What is the capital of France?",easy
   "Explain how enzymes lower activation energy.",medium
   "Critically evaluate the bias-variance trade-off in regularized models.",hard
   ```
   A JSON file with `[{"text": "...", "difficulty": "easy"}, ...]` also works.
3. Train:
   ```bash
   python train_difficulty.py --data training_data.csv --epochs 25
   ```
   This prints a real training loop (train/val loss, validation accuracy per
   epoch) and writes `difficulty_weights.pt` next to the scripts.
4. Restart `app.py` — it automatically picks up the fine-tuned weights.
   Check `/api/v1/ml-info` or the server logs to confirm it loaded.

Training runs on CPU in a few minutes even for a few thousand rows, because
only the small classifier head is trained — the sentence embedder itself
stays frozen (this is what makes CPU-only fine-tuning practical here).

## Notes

- If no internet is available on first run, `ml_engine.py` degrades
  gracefully to a rule-based fallback so the app keeps working, but you lose
  the neural question generation quality — internet is required at least once.
- PDF extraction requires text-based PDFs (not scanned images). Scanned PDFs
  would need OCR, which isn't included here.
- The `/health` endpoint reports exactly which mode each model loaded in,
  useful for debugging slow startups or fallback behavior.

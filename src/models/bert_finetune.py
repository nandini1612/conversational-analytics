"""
Phase 3 — DistilBERT Fine-Tuning
Person 2 (Models & Evaluation)

Designed to run on Google Colab with GPU.
Fine-tunes distilbert-base-uncased with a regression head on raw transcript text.

Expected finding: DistilBERT will likely underperform Ridge because the transcripts
are synthetic placeholder text ("sample billing_error conversation turn 3 uh").
This is a genuine finding — document it in the final report.

Outputs:
  models_saved/bert_weights/       ← fine-tuned model weights
  models_saved/bert_val_preds.npy  ← val predictions for Phase 4 ensemble

Usage (Colab):
  1. Mount Google Drive
  2. Upload train_features.csv and val_features.csv to Drive
  3. Run all cells
  4. Download bert_weights/ and bert_val_preds.npy to models_saved/

DO NOT load test_features.csv here — that is Phase 5 only.
"""

# ── Colab setup (uncomment when running on Colab) ────────────
# !pip install transformers torch pandas numpy scikit-learn

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────
# When running locally: ROOT = project root
# When running on Colab: adjust FEATURES_DIR and OUTPUTS_DIR below
try:
    ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Colab: __file__ not defined
    ROOT = Path("/content/drive/MyDrive/conv_analytics")

FEATURES_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR  = ROOT / "models"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

BERT_WEIGHTS_DIR     = OUTPUTS_DIR / "bert_weights"
BERT_VAL_PREDS_PATH  = ROOT / "outputs" / "predictions" / "bert_val_preds.npy"
RIDGE_VAL_PREDS_PATH = ROOT / "outputs" / "predictions" / "ridge_val_preds.npy"

# ── Hyperparameters ──────────────────────────────────────────
LR = 2e-5
BATCH_SIZE = 8
MAX_EPOCHS = 3
MAX_LENGTH = 512
RANDOM_STATE = 42


# ── Data loading ─────────────────────────────────────────────

def load_data():
    """Load train/val CSVs. Returns DataFrames with transcript_text and csat_score."""
    train_path = FEATURES_DIR / "train_features.csv"
    val_path = FEATURES_DIR / "val_features.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path}. Upload train_features.csv to Drive first."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Ensure transcript_text column exists
    if "transcript_text" not in train_df.columns:
        raise ValueError(
            "transcript_text column not found. Person 1 must include raw transcripts."
        )

    # Drop rows with missing transcript or label
    train_df = train_df.dropna(subset=["transcript_text", "csat_score"])
    val_df = val_df.dropna(subset=["transcript_text", "csat_score"])

    print(f"  Train: {len(train_df)} rows  Val: {len(val_df)} rows")
    return train_df, val_df


# ── Model ────────────────────────────────────────────────────

def build_model():
    """
    Load distilbert-base-uncased with a single-output regression head.
    num_labels=1 replaces the classification head with a regression head.
    """
    from transformers import DistilBertForSequenceClassification
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=1,
        problem_type="regression",
    )
    return model


# ── Dataset ──────────────────────────────────────────────────

def make_dataset(texts, labels, tokenizer):
    """Tokenize texts and return a PyTorch Dataset."""
    import torch
    from torch.utils.data import Dataset

    encodings = tokenizer(
        list(texts),
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    class CSATDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    return CSATDataset(encodings, labels)


# ── Training ─────────────────────────────────────────────────

def train(train_df, val_df, epochs=MAX_EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """
    Fine-tune DistilBERT on transcript text.
    Early stopping: stop if val loss does not improve for 1 consecutive epoch.

    Saves:
      models_saved/bert_weights/
      models_saved/bert_val_preds.npy
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import DistilBertTokenizer
    from sklearn.metrics import mean_absolute_error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cpu":
        print("  [WARN] Running on CPU — fine-tuning will be slow. Use Colab GPU.")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = build_model().to(device)

    train_dataset = make_dataset(
        train_df["transcript_text"].tolist(),
        train_df["csat_score"].tolist(),
        tokenizer,
    )
    val_dataset = make_dataset(
        val_df["transcript_text"].tolist(),
        val_df["csat_score"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    no_improve_count = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = outputs.logits.squeeze(-1).cpu().numpy()
                all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss /= len(val_loader)
        val_preds_clipped = np.clip(all_preds, 1.0, 5.0)
        val_mae = mean_absolute_error(all_labels, val_preds_clipped)

        print(f"  Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_MAE={val_mae:.4f}")

        # Early stopping: stop if val loss doesn't improve for 1 epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            # Save best weights
            model.save_pretrained(str(BERT_WEIGHTS_DIR))
            tokenizer.save_pretrained(str(BERT_WEIGHTS_DIR))
            best_val_preds = val_preds_clipped.copy()
            print(f"    → Saved best weights (epoch {epoch})")
        else:
            no_improve_count += 1
            print(f"    → No improvement ({no_improve_count}/1 patience)")
            if no_improve_count >= 1:
                print(f"  Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    # Save val predictions for Phase 4 ensemble
    np.save(str(BERT_VAL_PREDS_PATH), best_val_preds)
    print(f"\n  Saved val predictions → {BERT_VAL_PREDS_PATH}")

    # Compare with Ridge val MAE
    if RIDGE_VAL_PREDS_PATH.exists():
        try:
            ridge_preds = np.load(str(RIDGE_VAL_PREDS_PATH))
            # Load val labels for comparison
            val_df_reload = pd.read_csv(FEATURES_DIR / "val_features.csv")
            y_val = val_df_reload["csat_score"].values.astype(float)
            ridge_mae = mean_absolute_error(y_val[:len(ridge_preds)], ridge_preds)
            bert_mae = mean_absolute_error(all_labels, best_val_preds)
            if bert_mae > ridge_mae:
                print(f"\n  [FINDING] DistilBERT val MAE ({bert_mae:.4f}) > Ridge val MAE ({ridge_mae:.4f})")
                print("  → Expected. Synthetic transcripts give BERT no real language signal.")
                print("    Document this explicitly in the final report.")
        except Exception:
            pass

    print(f"\n  Saved weights → {BERT_WEIGHTS_DIR}")
    return best_val_preds


# ── Main ─────────────────────────────────────────────────────

def run_phase3():
    print("=" * 60)
    print("PHASE 3 — DISTILBERT FINE-TUNING")
    print("=" * 60)

    print("\n── Loading data ──")
    train_df, val_df = load_data()

    print("\n── Fine-tuning ──")
    val_preds = train(train_df, val_df)

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print(f"  Weights: {BERT_WEIGHTS_DIR}")
    print(f"  Val preds: {BERT_VAL_PREDS_PATH}")
    print("  Next: python src/models/ensemble.py")
    print("=" * 60)

    return val_preds


if __name__ == "__main__":
    run_phase3()

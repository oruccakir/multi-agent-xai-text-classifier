"""Transformer-based classifier for text classification."""

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_model import BaseModel
from .transformer_utils import (
    TextClassificationDataset,
    build_lora_config,
    get_lora_target_modules,
)


class TransformerClassifier(BaseModel):
    """
    Transformer-based classifier using HuggingFace transformers.

    Supports three fine-tuning strategies:
      - head_only: freeze base model, train classification head only
      - full: fine-tune all parameters
      - lora: parameter-efficient fine-tuning via LoRA adapters
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        fine_tune_mode: str = "head_only",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 3,
        max_length: int = 256,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
    ):
        super().__init__(name="Transformer")
        self.model_name = model_name
        self.num_labels = num_labels
        self.fine_tune_mode = fine_tune_mode
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay

        self.tokenizer = None
        self.model = None
        self.label2id = None
        self.id2label = None
        # Use provided device or auto-detect
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_model(self) -> None:
        """Initialize the transformer model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        if self.fine_tune_mode == "head_only":
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "pre_classifier" not in name:
                    param.requires_grad = False

        elif self.fine_tune_mode == "lora":
            from peft import get_peft_model

            target_modules = get_lora_target_modules(self.model_name)
            lora_config = build_lora_config(
                r=self.lora_r,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)

        # full mode: all parameters are trainable by default

        self.model.to(self.device)

    def fit(self, X_texts: List[str], y) -> "TransformerClassifier":
        """
        Fine-tune the transformer model on raw text data.

        Args:
            X_texts: List of raw text strings.
            y: Array-like of string labels.
        """
        y = np.asarray(y)
        unique_labels = sorted(set(y))
        self.classes_ = np.array(unique_labels)
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(unique_labels)

        self._initialize_model()

        y_encoded = np.array([self.label2id[label] for label in y])

        encodings = self.tokenizer(
            list(X_texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = TextClassificationDataset(encodings, y_encoded)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer
        if self.fine_tune_mode == "full":
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_params = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=self.learning_rate)
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

        # Scheduler with warmup
        total_steps = len(dataloader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"    Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    @torch.no_grad()
    def predict(self, X_texts) -> np.ndarray:
        """Predict class labels for raw texts."""
        proba = self.predict_proba(X_texts)
        indices = np.argmax(proba, axis=1)
        return np.array([self.id2label[i] for i in indices])

    @torch.no_grad()
    def predict_proba(self, X_texts) -> np.ndarray:
        """Return prediction probabilities for raw texts."""
        self.model.eval()
        all_probs = []

        texts = list(X_texts)
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    def evaluate(self, X_texts, y) -> dict:
        """Evaluate the model on test data (accepts raw texts)."""
        y_pred = self.predict(X_texts)
        y = np.asarray(y)

        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        return {
            "model_name": self.name,
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        }

    def save(self, path: str) -> None:
        """Save the transformer model to a directory."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        model_to_save = self.model
        if self.fine_tune_mode == "lora":
            # Save the full merged model for easy loading
            model_to_save = self.model.merge_and_unload()

        model_to_save.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save metadata
        meta = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "fine_tune_mode": self.fine_tune_mode,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "max_length": self.max_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
        }
        with open(save_dir / "transformer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TransformerClassifier":
        """Load a transformer model from a directory.
        
        Args:
            path: Path to the saved model directory.
            device: Device to load model onto (e.g., 'cpu', 'cuda:0').
        """
        load_dir = Path(path)

        with open(load_dir / "transformer_meta.json") as f:
            meta = json.load(f)

        instance = cls(
            model_name=meta["model_name"],
            num_labels=meta["num_labels"],
            fine_tune_mode=meta["fine_tune_mode"],
            learning_rate=meta.get("learning_rate", 2e-5),
            batch_size=meta.get("batch_size", 16),
            epochs=meta.get("epochs", 3),
            max_length=meta.get("max_length", 256),
            lora_r=meta.get("lora_r", 8),
            lora_alpha=meta.get("lora_alpha", 16),
            lora_dropout=meta.get("lora_dropout", 0.1),
            warmup_ratio=meta.get("warmup_ratio", 0.1),
            weight_decay=meta.get("weight_decay", 0.01),
            device=device,
        )

        instance.tokenizer = AutoTokenizer.from_pretrained(str(load_dir))
        instance.model = AutoModelForSequenceClassification.from_pretrained(str(load_dir))
        instance.model.to(instance.device)
        instance.model.eval()

        instance.label2id = meta["label2id"]
        instance.id2label = {int(k): v for k, v in meta["id2label"].items()}
        instance.classes_ = np.array(meta["classes"]) if meta.get("classes") else None
        instance.is_fitted = True

        print(f"Model loaded from {load_dir}")
        return instance

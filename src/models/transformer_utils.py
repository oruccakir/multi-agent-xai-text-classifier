"""Utility classes and functions for transformer-based classification."""

import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for tokenized text classification."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Mapping from model architecture to LoRA target module names
_LORA_TARGET_MODULES = {
    "bert": ["query", "value"],
    "distilbert": ["q_lin", "v_lin"],
    "roberta": ["query", "value"],
    "electra": ["query", "value"],
    "albert": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "deberta": ["query_proj", "value_proj"],
}


def get_lora_target_modules(model_name: str) -> list[str]:
    """Return the LoRA target module names for a given HuggingFace model."""
    model_name_lower = model_name.lower()
    for key, modules in _LORA_TARGET_MODULES.items():
        if key in model_name_lower:
            return modules
    # Default fallback for unknown architectures
    return ["query", "value"]


def build_lora_config(r: int, alpha: int, dropout: float, target_modules: list[str]):
    """Build a peft LoraConfig for sequence classification."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )

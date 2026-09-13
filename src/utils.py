"""Utility functions for model loading, data preparation, and experiment helpers."""

import os
from typing import List, Tuple, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset


def load_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load a pretrained causal language model.

    Args:
        model_name: Hugging Face model identifier (e.g., 'gpt2', 'facebook/opt-125m').
        cache_dir: Optional directory to cache the model.
        device: Target device. Defaults to CPU if not specified.

    Returns:
        The loaded model in evaluation mode.
    """
    if device is None:
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    return model


def load_calibration_data(
    dataset_name: str = 'wikitext',
    split: str = 'train',
    max_samples: Optional[int] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load and tokenize calibration data from a Hugging Face dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'wikitext', 'ptb_text_only').
        split: Dataset split to use.
        max_samples: Maximum number of samples to return.

    Returns:
        List of (input_ids, labels) tuples.
    """
    dataset = load_dataset(dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = []
    for item in dataset:
        text = item.get('text', item.get('sentence', str(item)))
        encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()
        samples.append((input_ids, labels))

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def get_random_examples(
    dataloader: List,
    num_samples: int
) -> List:
    """
    Randomly sample calibration examples from a dataloader.

    Args:
        dataloader: List of calibration samples.
        num_samples: Number of samples to select.

    Returns:
        List of randomly selected samples.
    """
    import random
    random.seed(42)  # Fixed seed for reproducibility

    if num_samples >= len(dataloader):
        return dataloader

    indices = random.sample(range(len(dataloader)), num_samples)
    return [dataloader[i] for i in indices]


def get_selected_examples(
    dataloader: List,
    num_samples: int,
    selection_method: str = 'entropy'
) -> List:
    """
    Select calibration examples based on a selection criterion.

    Args:
        dataloader: List of calibration samples.
        num_samples: Number of samples to select.
        selection_method: Method for selection ('entropy', 'loss', 'random').

    Returns:
        List of selected samples.
    """
    import random
    random.seed(42)

    if selection_method == 'random':
        return get_random_examples(dataloader, num_samples)

    # For entropy-based selection, we'd need model logits
    # Placeholder for now - returns random for simplicity
    return get_random_examples(dataloader, num_samples)


def save_results(results: dict, output_path: str):
    """Save experiment results to a JSON file."""
    import json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(output_path: str) -> dict:
    """Load experiment results from a JSON file."""
    import json
    with open(output_path, 'r') as f:
        return json.load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_subset(
    dataset: Dataset,
    num_samples: int,
    method: str = 'random'
) -> Dataset:
    """Create a subset of a dataset for calibration."""
    if method == 'random':
        return dataset.select(range(min(num_samples, len(dataset))))
    elif method == 'first':
        return dataset.select(range(min(num_samples, len(dataset))))
    else:
        raise ValueError(f"Unknown selection method: {method}")

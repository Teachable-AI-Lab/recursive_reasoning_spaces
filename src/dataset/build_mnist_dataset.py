from typing import Optional
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from torchvision import datasets

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/mnist"
    data_dir: str = "data/.raw"
    subsample_train: Optional[int] = None
    subsample_test: Optional[int] = None
    seed: int = 0


def _subsample(inputs: np.ndarray, labels: np.ndarray, size: Optional[int], rng: np.random.Generator):
    if size is None or size >= inputs.shape[0]:
        return inputs, labels

    indices = rng.choice(inputs.shape[0], size=size, replace=False)
    return inputs[indices], labels[indices]


def _build_subset(inputs_flat: np.ndarray, labels: np.ndarray):
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}

    seq_len = 1 + 28 * 28

    # Token IDs
    # 0: PAD / IGNORE
    # 1..256: grayscale pixel value 0..255
    # 257..266: class labels 0..9
    # 267: query token
    query_token = 267

    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for pixels, label in zip(inputs_flat, labels):
        inp = np.empty((seq_len,), dtype=np.int32)
        inp[0] = query_token
        inp[1:] = pixels.astype(np.int32) + 1

        out = np.zeros((seq_len,), dtype=np.int32)
        out[0] = int(label) + 257

        results["inputs"].append(inp)
        results["labels"].append(out)

        example_id += 1
        puzzle_id += 1
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)

    return {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }


def _save_subset(output_dir: str, split: str, data: dict):
    metadata = PuzzleDatasetMetadata(
        seq_len=1 + 28 * 28,
        vocab_size=268,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(data["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(data["group_indices"]) - 1,
        sets=["all"],
    )

    save_dir = os.path.join(output_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for key, value in data.items():
        np.save(os.path.join(save_dir, f"all__{key}.npy"), value)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)

    train_ds = datasets.MNIST(root=config.data_dir, train=True, download=True)
    test_ds = datasets.MNIST(root=config.data_dir, train=False, download=True)

    train_inputs = train_ds.data.numpy().reshape(-1, 28 * 28)
    test_inputs = test_ds.data.numpy().reshape(-1, 28 * 28)
    train_labels = train_ds.targets.numpy()
    test_labels = test_ds.targets.numpy()

    train_inputs, train_labels = _subsample(train_inputs, train_labels, config.subsample_train, rng)
    test_inputs, test_labels = _subsample(test_inputs, test_labels, config.subsample_test, rng)

    train_data = _build_subset(train_inputs, train_labels)
    test_data = _build_subset(test_inputs, test_labels)

    _save_subset(config.output_dir, "train", train_data)
    _save_subset(config.output_dir, "test", test_data)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()
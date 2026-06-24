"""
Generate non-IID synthetic data partitions for E2E testing.

Creates 5 client data partitions with overlapping class distributions
to simulate non-IID federated learning, plus a shared test set.

Usage:
    python scripts/synth_data.py              # generates data/ dir
    python scripts/synth_data.py --num-clients 3 --samples-per-client 300

Output:
    scripts/data/client_{i}_data.pt   — (X: Tensor[N,784], y: Tensor[N])
    scripts/data/test_data.pt         — (X: Tensor[200,784], y: Tensor[200])
"""

import argparse
import os

import torch


def generate_partition(
    client_id: int,
    num_clients: int = 5,
    num_samples: int = 200,
    num_classes: int = 10,
    input_dim: int = 784,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a non-IID data partition for one client.

    Each client gets samples biased toward a sliding window of classes.
    With 5 clients and 10 classes:
      client 0 → classes {0,1,2}
      client 1 → classes {2,3,4}
      client 2 → classes {4,5,6}
      client 3 → classes {6,7,8}
      client 4 → classes {8,9,0}
    """
    rng = torch.Generator().manual_seed(seed + client_id)

    classes_per_client = max(2, num_classes // num_clients + 1)
    start_class = (client_id * 2) % num_classes
    primary_classes = [(start_class + c) % num_classes for c in range(classes_per_client)]

    samples_per_class = num_samples // len(primary_classes)
    remainder = num_samples - samples_per_class * len(primary_classes)

    X_parts = []
    y_parts = []
    for i, cls in enumerate(primary_classes):
        n = samples_per_class + (1 if i < remainder else 0)
        # Class-specific centroid in feature space (so classes are separable)
        centroid = torch.randn(input_dim, generator=rng) * 0.3
        # Shift centroid by class index so classes are distinct
        centroid += cls * 0.1
        data = centroid + torch.randn(n, input_dim, generator=rng) * 0.15
        labels = torch.full((n,), cls, dtype=torch.long)
        X_parts.append(data)
        y_parts.append(labels)

    X = torch.cat(X_parts)
    y = torch.cat(y_parts)

    # Shuffle
    perm = torch.randperm(len(X), generator=rng)
    return X[perm], y[perm]


def generate_test_set(
    num_samples: int = 200,
    num_classes: int = 10,
    input_dim: int = 784,
    seed: int = 9999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a balanced test set with uniform class distribution."""
    rng = torch.Generator().manual_seed(seed)
    samples_per_class = num_samples // num_classes
    remainder = num_samples - samples_per_class * num_classes

    X_parts = []
    y_parts = []
    for cls in range(num_classes):
        n = samples_per_class + (1 if cls < remainder else 0)
        centroid = torch.randn(input_dim, generator=rng) * 0.3 + cls * 0.1
        data = centroid + torch.randn(n, input_dim, generator=rng) * 0.15
        labels = torch.full((n,), cls, dtype=torch.long)
        X_parts.append(data)
        y_parts.append(labels)

    X = torch.cat(X_parts)
    y = torch.cat(y_parts)
    perm = torch.randperm(len(X), generator=rng)
    return X[perm], y[perm]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic non-IID data for ASTRA E2E tests")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=200)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_clients} client partitions ({args.samples_per_client} samples each)...")
    for i in range(args.num_clients):
        X, y = generate_partition(
            i,
            num_clients=args.num_clients,
            num_samples=args.samples_per_client,
            num_classes=args.num_classes,
            input_dim=args.input_dim,
        )
        path = os.path.join(args.output_dir, f"client_{i}_data.pt")
        torch.save({"X": X, "y": y}, path)
        # Show class distribution
        unique, counts = y.unique(return_counts=True)
        dist = {int(c): int(n) for c, n in zip(unique, counts)}
        print(f"  Client {i}: {len(X)} samples, classes = {dist}")

    print(f"Generating test set (200 samples, balanced)...")
    X_test, y_test = generate_test_set(
        num_samples=200,
        num_classes=args.num_classes,
        input_dim=args.input_dim,
    )
    test_path = os.path.join(args.output_dir, "test_data.pt")
    torch.save({"X": X_test, "y": y_test}, test_path)
    unique, counts = y_test.unique(return_counts=True)
    dist = {int(c): int(n) for c, n in zip(unique, counts)}
    print(f"  Test: {len(X_test)} samples, classes = {dist}")

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from lsr.lsr import get_scene_image_paths, load_batch_images
from lsr.nnet import NNetwork


def load_checkpoints(model: NNetwork, checkpoint_path: str):
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path))

    return model


def train(
    model: NNetwork,
    data_dir: str,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    batch_size: int = 32,
):
    if checkpoint_path is not None:
        model = load_checkpoints(model, checkpoint_path)
    model = model.to(device)
    model.train()

    # Load all image paths
    scenes = get_scene_image_paths(data_dir)
    all_paths = [p for paths in scenes.values() for p in paths]
    print(f"Found {len(all_paths)} images for training")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_start in range(0, len(all_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(all_paths))
            batch_paths = all_paths[batch_start:batch_end]

            # Load batch of images
            batch = load_batch_images(batch_paths, size=(150, 150)).to(device)

            # Pseudo-labels (random classification task to learn features)
            labels = torch.randint(0, model.fc2.out_features, (batch.size(0),)).to(
                device
            )

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the network")

    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="ckpts/lsr.pt",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    model = NNetwork()
    train(
        model,
        args.data_dir,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
    )

    # Ensure checkpoint directory exists
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

"""
Load the model trained by train.py and run inference. No distributed setup —
just verifying the saved weights work standalone.
"""
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    weights = os.path.join(HERE, "model.pt")
    if not os.path.exists(weights):
        print(f"no weights at {weights}; run train.py first")
        sys.exit(1)

    model = MLP(hidden=64)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()

    xs = torch.linspace(0, 1, 21).unsqueeze(1)
    with torch.no_grad():
        ys = model(xs).squeeze()
    truth = torch.sin(2 * math.pi * xs.squeeze())

    print(f"  {'x':>5} {'pred':>10} {'sin(2pi*x)':>12} {'err':>10}")
    for x, p, t in zip(xs.squeeze(), ys, truth):
        print(f"  {x.item():.3f}  {p.item():>+9.4f}    {t.item():>+9.4f}  "
              f"{abs(p.item() - t.item()):>+9.4f}")
    rmse = ((ys - truth) ** 2).mean().sqrt().item()
    print(f"\n  rmse on test grid: {rmse:.4f}")


if __name__ == "__main__":
    main()

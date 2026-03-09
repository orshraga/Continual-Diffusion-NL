from typing import Dict, Tuple
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

from network import DDPM, ContextUnet


def predict_mnist():
    # -----------------------------
    # Hyperparameters (must match training)
    # -----------------------------
    n_T = 400
    device = "cuda:0"  # change to "cpu" if needed
    n_classes = 10
    n_feat = 128
    n_sample = 5

    # -----------------------------
    # Paths
    # -----------------------------
    base_dir = "/content/drive/MyDrive/Diffusion_Model/full_baseline/output"
    os.makedirs(base_dir, exist_ok=True)

    # Folder for prediction images
    save_dir = os.path.join(base_dir, "prediction")
    os.makedirs(save_dir, exist_ok=True)

    # Choose which trained model to load (e.g. last epoch 19)
    model_path = os.path.join(base_dir, "model_19.pth")

    # -----------------------------
    # Model
    # -----------------------------
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
    )
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.to(device)
    ddpm.eval()

    # -----------------------------
    # Data loader (just to grab some labels)
    # -----------------------------
    tf = transforms.Compose([transforms.ToTensor()])  # MNIST is [0,1]
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=n_sample, shuffle=True, num_workers=5)

    # -----------------------------
    # Sampling
    # -----------------------------
    with torch.no_grad():
        x, c = next(iter(dataloader))
        x = x.to(device)
        c = c.to(device)

        # Generate conditioned samples
        x_gen, x_gen_store = ddpm.prediction(
            n_sample, (1, 28, 28), c, device, guide_w=2.0
        )

        # Concatenate generated + real for visualization
        x_all = torch.cat([x_gen, x])
        grid = make_grid(x_all * -1 + 1, nrow=n_sample)

        out_path = os.path.join(save_dir, "prediction.png")
        save_image(grid, out_path)
        print("saved image at " + out_path)


if __name__ == "__main__":
    predict_mnist()

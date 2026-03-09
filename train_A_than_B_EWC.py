from typing import Iterable, List, Optional, Dict, Any, Tuple
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from scipy import io
from tqdm import tqdm
import os
from network import DDPM, ContextUnet


def _subset_mnist_by_digits(dataset: MNIST, digits: Iterable[int]) -> Subset:
    """
    Returns a torch.utils.data.Subset of MNIST containing only the requested digit labels.
    Keeps the ORIGINAL labels (0..9) because your ContextUnet expects class ids in [0..n_classes-1].
    """
    digits_set = set(int(d) for d in digits)

    # MNIST stores labels in dataset.targets (torch tensor)
    targets = dataset.targets
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for d in digits_set:
        mask |= (targets == d)

    idx = mask.nonzero(as_tuple=False).squeeze(1).tolist()
    return Subset(dataset, idx)

def compute_fisher(ddpm_model, dataset, device, num_samples=200):
    """
    Computes the Fisher Information Matrix (diagonal approximation) for EWC.
    """
    fisher = {}
    params_opt = {}
    
    # 1. Store the "star" parameters (weights from Task A)
    for n, p in ddpm_model.named_parameters():
        params_opt[n] = p.data.clone().detach()
        fisher[n] = torch.zeros_like(p)

    # 2. Compute gradients on a subset of data
    ddpm_model.eval()
    # Create a small loader just for Fisher calculation
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Computing Fisher Information...")
    count = 0
    for x, c in loader:
        ddpm_model.zero_grad()
        x = x.to(device)
        c = c.to(device)
        
        # Forward pass (get loss)
        loss = ddpm_model(x, c)
        loss.backward()
        
        # Accumulate squared gradients
        for n, p in ddpm_model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.clone() ** 2
        
        count += len(x)
        if count >= num_samples:
            break
            
    # 3. Normalize
    for n in fisher:
        fisher[n] /= count
        
    return fisher, params_opt

def _load_weights_into_ddpm(ddpm: DDPM, path: str, device: str) -> Dict[str, Any]:
    """
    Loads either:
      - a checkpoint dict with 'model_state_dict'
      - or a raw state_dict
    Returns a small metadata dict about what was loaded.
    """
    obj = torch.load(path, map_location=device)

    meta: Dict[str, Any] = {"loaded_from": path}
    if isinstance(obj, dict) and "model_state_dict" in obj:
        ddpm.load_state_dict(obj["model_state_dict"])
        meta["type"] = "checkpoint"
        meta["epoch"] = obj.get("epoch", None)
        meta["n_T"] = obj.get("n_T", None)
        meta["n_classes"] = obj.get("n_classes", None)
        meta["n_feat"] = obj.get("n_feat", None)
        meta["digits"] = obj.get("digits", None)
        meta["phase"] = obj.get("phase", None)
    else:
        ddpm.load_state_dict(obj)
        meta["type"] = "state_dict"

    return meta


def train_on_digits(
    *,
    digits: List[int],
    save_dir: str,
    # Hyperparameters (defaults match your existing file)
    early_stop_loss: float = 0.025,
    n_epoch: int = 100,
    batch_size: int = 256,
    n_T: int = 400,
    n_classes: int = 10,
    n_feat: int = 128,
    lrate: float = 1e-4,
    num_workers: int = 5,
    # Loading
    pretrained_path: Optional[str] = None,   # can be .pth state_dict OR .pt checkpoint
    # Saving
    model_out_name: str = "model_final.pth",
    save_checkpoints_each_epoch: bool = True,   # saves checkpoint_epoch_XXXX.pt
    # EWC   
    ewc_fisher: Optional[Dict] = None,   # The Matrix from Task A
    ewc_params: Optional[Dict] = None,   # The Weights from Task A
    ewc_lambda: float = 0.0,             # Strength (e.g., 400)
) -> str:
    """
    Trains the diffusion model ONLY on the specified digits, with early stopping.
    Saves:
      - checkpoint_last.pt (every epoch)
      - optional checkpoint_epoch_XXXX.pt
      - loss.mat (step_ema + epoch_mean)
      - config.json
      - model_out_name (on last epoch or early stop)
    Returns: path to the saved final model (state_dict).
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_dir = str(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Print parameters (like your existing train_mnist)
    # -----------------------------
    print("========== TRAINING (train_on_digits) ==========")
    print(f"digits              : {digits}")
    print(f"pretrained_path     : {pretrained_path}")
    print(f"early_stop_loss     : {early_stop_loss}")
    print(f"n_epoch             : {n_epoch}")
    print(f"batch_size          : {batch_size}")
    print(f"n_T                 : {n_T}")
    print(f"device              : {device}")
    print(f"n_classes           : {n_classes}")
    print(f"n_feat              : {n_feat}")
    print(f"lrate               : {lrate}")
    print(f"num_workers         : {num_workers}")
    print(f"save_dir            : {save_dir}")
    print("===============================================")

    # -----------------------------
    # Model
    # -----------------------------
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
    ).to(device)

    load_meta = None
    if pretrained_path is not None:
        load_meta = _load_weights_into_ddpm(ddpm, pretrained_path, device)
        print("[train_on_digits] Loaded pretrained weights:", load_meta)

    # -----------------------------
    # Data (filtered by digits)
    # -----------------------------
    tf = transforms.Compose([transforms.ToTensor()])
    full_dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataset = _subset_mnist_by_digits(full_dataset, digits)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # -----------------------------
    # Optimizer + scheduler (same idea as your existing file)
    # -----------------------------
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda epoch: 1 - epoch / n_epoch if epoch < n_epoch else 0.0,
    )

    # logging
    loss_step_ema: List[float] = []
    loss_epoch_mean: List[float] = []

    # -----------------------------
    # Save config for reproducibility
    # -----------------------------
    cfg = {
        "digits": digits,
        "pretrained_path": pretrained_path,
        "load_meta": load_meta,
        "early_stop_loss": early_stop_loss,
        "n_epoch": n_epoch,
        "batch_size": batch_size,
        "n_T": n_T,
        "n_classes": n_classes,
        "n_feat": n_feat,
        "lrate": lrate,
        "num_workers": num_workers,
        "device": device,
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # -----------------------------
    # Training loop
    # -----------------------------
    for ep in range(n_epoch):
        print(f"epoch {ep}")
        ddpm.train()

        pbar = tqdm(dataloader)
        ema = None
        epoch_losses = []

        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)

            loss = ddpm(x, c)

            # === EWC LOGIC START ===
            if ewc_fisher is not None and ewc_params is not None:
                ewc_loss = 0.0
                for n, p in ddpm.named_parameters():
                    # Formula: F * (theta - theta_star)^2
                    # We match parameter names to find the right Fisher value
                    if n in ewc_fisher:
                        f_val = ewc_fisher[n].to(device)
                        p_star = ewc_params[n].to(device)
                        ewc_loss += (f_val * (p - p_star) ** 2).sum()
                
                # Equation 3 from EWC paper: Loss = L_B + (lambda/2) * Penalty
                loss = loss + (ewc_lambda / 2.0) * ewc_loss
            # === EWC LOGIC END ===

            loss.backward()
            optim.step()

            l = float(loss.item())
            epoch_losses.append(l)

            if ema is None:
                ema = l
            else:
                ema = 0.95 * ema + 0.05 * l

            pbar.set_description(f"loss_ema: {ema:.4f}")

            loss_step_ema.append(ema)

        scheduler.step()

        # epoch mean
        epoch_mean = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        loss_epoch_mean.append(epoch_mean)

        # -----------------------------
        # Save checkpoint every epoch (like your current checkpoint_last.pt)
        # -----------------------------
        checkpoint = {
            "phase": "train_on_digits",
            "digits": digits,
            "epoch": ep,
            "model_state_dict": ddpm.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "n_T": n_T,
            "n_classes": n_classes,
            "n_feat": n_feat,
            "loss_epoch_mean": loss_epoch_mean,
        }
        ckpt_last = os.path.join(save_dir, "checkpoint_last.pt")
        torch.save(checkpoint, ckpt_last)
        print(f"saved checkpoint at {ckpt_last}")

        if save_checkpoints_each_epoch:
            ckpt_ep = os.path.join(save_dir, f"checkpoint_epoch_{ep:04d}.pt")
            torch.save(checkpoint, ckpt_ep)

        # -----------------------------
        # Save model+loss if last epoch or early stop
        # -----------------------------
        did_early_stop = (ema is not None) and (ema < early_stop_loss)
        is_last_epoch = (ep == n_epoch - 1)

        if did_early_stop:
            print(f"[EARLY STOP] Stopping training because loss_ema={ema:.6f} < {early_stop_loss:.6f}")

        if is_last_epoch or did_early_stop:
            model_path = os.path.join(save_dir, model_out_name)
            torch.save(ddpm.state_dict(), model_path)
            print("saved model at " + model_path)

            loss_path = os.path.join(save_dir, "loss.mat")
            io.savemat(
                loss_path,
                {"loss_step_ema": loss_step_ema, "loss_epoch_mean": loss_epoch_mean},
            )
            print("saved loss at " + loss_path)

            return model_path  # IMPORTANT: return the final model path

    # Should never reach here, but just in case:
    model_path = os.path.join(save_dir, model_out_name)
    torch.save(ddpm.state_dict(), model_path)
    return model_path


def train_two_phase_continual(
    *,
    # Phase 1 / Task A
    digits_A: List[int] = [0, 1, 2, 3, 4, 5],
    # Phase 2 / Task B
    digits_B: List[int] = [6, 7, 8, 9],
    # Shared hparams
    early_stop_loss: float = 0.025,
    n_epoch: int = 100,
    batch_size: int = 256,
    n_T: int = 400,
    n_classes: int = 10,
    n_feat: int = 128,
    lrate: float = 1e-4,
    num_workers: int = 5,
) -> Tuple[str, str]:
    """
    Produces exactly the two models you asked for:
      1) Model trained on Task A only (digits 0–5)
      2) Model trained on Task A, then continued on Task B only (digits 6–9)

    This matches your project plan: Phase 1 on 0–5, Phase 2 on 6–9 (no access to 0–5). 
    """
    save_root = os.environ.get("DM_OUTPUT_DIR", "./output")
    os.makedirs(save_root, exist_ok=True)
    print("[train_two_phase_continual] Saving output to:", save_root)

    # Put each phase in its own folder inside DM_OUTPUT_DIR
    phaseA_dir = os.path.join(save_root, "phase1_taskA_digits_0to5")
    phaseB_dir = os.path.join(save_root, "phase2_taskB_digits_6to9_from_taskA")
    os.makedirs(phaseA_dir, exist_ok=True)
    os.makedirs(phaseB_dir, exist_ok=True)

    # -----------------------------
    # Phase 1: train on Task A (0–5)
    # -----------------------------
    model_A_path = train_on_digits(
        digits=digits_A,
        save_dir=phaseA_dir,
        early_stop_loss=early_stop_loss,
        n_epoch=n_epoch,
        batch_size=batch_size,
        n_T=n_T,
        n_classes=n_classes,
        n_feat=n_feat,
        lrate=lrate,
        num_workers=num_workers,
        pretrained_path=None,
        model_out_name="model_taskA.pth",
        save_checkpoints_each_epoch=True,
    )

    # -----------------------------
    # INTERLUDE: Compute Fisher on Phase 1 Model
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the Task A model temporarily
    ddpm_A = DDPM(
        nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1
    ).to(device)
    ddpm_A.load_state_dict(torch.load(model_A_path, map_location=device))
    
    # 2. Load Task A data again for Fisher calc
    tf = transforms.Compose([transforms.ToTensor()])
    full_ds = MNIST("./data", train=True, download=True, transform=tf)
    ds_A = _subset_mnist_by_digits(full_ds, digits_A)
    
    # 3. Compute Matrix
    fisher_matrix, opt_params = compute_fisher(ddpm_A, ds_A, device)
    
    # Free up memory
    del ddpm_A
    torch.cuda.empty_cache()

    # -----------------------------
    # Phase 2: continue on Task B (6–9), starting from model A
    # -----------------------------
    model_A_then_B_path = train_on_digits(
        digits=digits_B,
        save_dir=phaseB_dir,
        early_stop_loss=early_stop_loss,
        n_epoch=n_epoch,
        batch_size=batch_size,
        n_T=n_T,
        n_classes=n_classes,
        n_feat=n_feat,
        lrate=lrate,
        num_workers=num_workers,
        pretrained_path=model_A_path,  # <-- continue from Task A
        model_out_name="model_taskA_then_B.pth",
        save_checkpoints_each_epoch=True,
        # PASS EWC ARGUMENTS HERE
        ewc_fisher=fisher_matrix,
        ewc_params=opt_params,
        ewc_lambda=400.0, # Value from paper (Table 2) [cite: 404]
    )

    print("\n=== DONE ===")
    print("Model after Task A only      :", model_A_path)
    print("Model after Task A then Task B:", model_A_then_B_path)

    return model_A_path, model_A_then_B_path


if __name__ == "__main__":
    # Two-phase continual training (Phase 1: 0–5, Phase 2: 6–9)
    train_two_phase_continual()

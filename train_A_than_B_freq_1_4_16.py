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


def _unique_params_from_modules(modules: List[torch.nn.Module]) -> List[torch.nn.Parameter]:
    """Collects parameters from modules, removing duplicates while preserving order."""
    seen: set[int] = set()
    params: List[torch.nn.Parameter] = []
    for m in modules:
        for p in m.parameters(recurse=True):
            pid = id(p)
            if pid not in seen:
                params.append(p)
                seen.add(pid)
    return params


def _split_contextunet_params_by_depth(unet: ContextUnet) -> Dict[str, List[torch.nn.Parameter]]:
    """
    Splits ContextUnet (network.py) parameters into three depth groups.

    For your current architecture:
      Encoder depths:
        - shallow: init_conv (28x28)
        - middle : down1 (28->14)
        - deep   : down2 + to_vec (14->7 + bottleneck)
      Decoder depths:
        - deep   : up0 (bottleneck -> 7x7) + (time/context embed1)
        - middle : up1 (7->14) + (time/context embed2)
        - shallow: up2 (14->28) + out head

    Returns dict with keys: 'shallow', 'middle', 'deep'.
    """
    shallow_mods = [unet.init_conv, unet.up2, unet.out]
    middle_mods = [unet.down1, unet.up1, unet.timeembed2, unet.contextembed2]
    deep_mods = [unet.down2, unet.to_vec, unet.up0, unet.timeembed1, unet.contextembed1]

    shallow_params = _unique_params_from_modules(shallow_mods)
    middle_params = _unique_params_from_modules(middle_mods)
    deep_params = _unique_params_from_modules(deep_mods)

    # Safety: ensure no overlaps
    shallow_ids = set(id(p) for p in shallow_params)
    middle_ids = set(id(p) for p in middle_params)
    deep_ids = set(id(p) for p in deep_params)

    overlap_sm = shallow_ids & middle_ids
    overlap_sd = shallow_ids & deep_ids
    overlap_md = middle_ids & deep_ids
    if overlap_sm or overlap_sd or overlap_md:
        raise RuntimeError(
            "Parameter overlap detected between depth groups. "
            "Check grouping logic in _split_contextunet_params_by_depth()."
        )

    return {"shallow": shallow_params, "middle": middle_params, "deep": deep_params}


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
    save_checkpoints_each_epoch: bool = False,   # saves checkpoint_epoch_XXXX.pt
    # Frequency-based / Nested-Learning style training
    # update_frequency = [shallow_freq, middle_freq, deep_freq]
    # Example: [1, 4, 16]
    update_frequency: Optional[List[int]] = None,
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
    print(f"update_frequency    : {update_frequency}")
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
    # Optimizer(s) + scheduler(s)
    # -----------------------------
    # If update_frequency is provided, we split the UNet into depth groups
    # and only step each group's optimizer every k batches.
    # Otherwise, we fall back to the standard "all layers every batch" training.
    use_frequency_updates = update_frequency is not None

    if use_frequency_updates:
        if len(update_frequency) != 3:
            raise ValueError(
                f"update_frequency must have length 3 [shallow, middle, deep], got {update_frequency}"
            )
        freq_shallow, freq_middle, freq_deep = [int(x) for x in update_frequency]
        if freq_shallow <= 0 or freq_middle <= 0 or freq_deep <= 0:
            raise ValueError(f"update_frequency must be positive ints, got {update_frequency}")

        # Depth groups based on ContextUnet modules in network.py
        groups = _split_contextunet_params_by_depth(ddpm.nn_model)
        shallow_params = groups["shallow"]
        middle_params = groups["middle"]
        deep_params = groups["deep"]

        opt_shallow = torch.optim.Adam(shallow_params, lr=lrate)
        opt_middle = torch.optim.Adam(middle_params, lr=lrate)
        opt_deep = torch.optim.Adam(deep_params, lr=lrate)

        sch_shallow = torch.optim.lr_scheduler.LambdaLR(
            opt_shallow,
            lr_lambda=lambda epoch: 1 - epoch / n_epoch if epoch < n_epoch else 0.0,
        )
        sch_middle = torch.optim.lr_scheduler.LambdaLR(
            opt_middle,
            lr_lambda=lambda epoch: 1 - epoch / n_epoch if epoch < n_epoch else 0.0,
        )
        sch_deep = torch.optim.lr_scheduler.LambdaLR(
            opt_deep,
            lr_lambda=lambda epoch: 1 - epoch / n_epoch if epoch < n_epoch else 0.0,
        )

        print("[train_on_digits] Frequency-based updates enabled.")
        print(f"  update_frequency (shallow/middle/deep): {update_frequency}")
        print(f"  shallow params: {sum(p.numel() for p in shallow_params):,}")
        print(f"  middle  params: {sum(p.numel() for p in middle_params):,}")
        print(f"  deep    params: {sum(p.numel() for p in deep_params):,}")
    else:
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
        "update_frequency": update_frequency,
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
    def _avg_grads_(params: List[torch.nn.Parameter], div: int) -> None:
        if div <= 1:
            return
        for p in params:
            if p.grad is not None:
                p.grad.div_(div)

    for ep in range(n_epoch):
        print(f"epoch {ep}")
        ddpm.train()

        pbar = tqdm(dataloader)
        ema = None
        epoch_losses = []

        if use_frequency_updates:
            # Important: do NOT zero middle/deep grads each batch.
            # We accumulate gradients until their update frequency triggers a step.
            opt_shallow.zero_grad(set_to_none=True)
            opt_middle.zero_grad(set_to_none=True)
            opt_deep.zero_grad(set_to_none=True)
            accum_s = 0
            accum_m = 0
            accum_d = 0

        for x, c in pbar:
            if not use_frequency_updates:
                optim.zero_grad(set_to_none=True)

            x = x.to(device)
            c = c.to(device)

            loss = ddpm(x, c)
            loss.backward()

            if use_frequency_updates:
                # We always accumulate gradients for all groups.
                accum_s += 1
                accum_m += 1
                accum_d += 1

                # Step shallow group
                if accum_s >= freq_shallow:
                    _avg_grads_(shallow_params, accum_s)
                    opt_shallow.step()
                    opt_shallow.zero_grad(set_to_none=True)
                    accum_s = 0

                # Step middle group
                if accum_m >= freq_middle:
                    _avg_grads_(middle_params, accum_m)
                    opt_middle.step()
                    opt_middle.zero_grad(set_to_none=True)
                    accum_m = 0

                # Step deep group
                if accum_d >= freq_deep:
                    _avg_grads_(deep_params, accum_d)
                    opt_deep.step()
                    opt_deep.zero_grad(set_to_none=True)
                    accum_d = 0
            else:
                optim.step()

            l = float(loss.item())
            epoch_losses.append(l)

            if ema is None:
                ema = l
            else:
                ema = 0.95 * ema + 0.05 * l

            pbar.set_description(f"loss_ema: {ema:.4f}")

            loss_step_ema.append(ema)

        # End of epoch: if we are accumulating, flush any remaining grads so
        # epoch checkpoints correspond to fully-applied optimizer states.
        if use_frequency_updates:
            if accum_s > 0:
                _avg_grads_(shallow_params, accum_s)
                opt_shallow.step()
                opt_shallow.zero_grad(set_to_none=True)
                accum_s = 0
            if accum_m > 0:
                _avg_grads_(middle_params, accum_m)
                opt_middle.step()
                opt_middle.zero_grad(set_to_none=True)
                accum_m = 0
            if accum_d > 0:
                _avg_grads_(deep_params, accum_d)
                opt_deep.step()
                opt_deep.zero_grad(set_to_none=True)
                accum_d = 0

            sch_shallow.step()
            sch_middle.step()
            sch_deep.step()
        else:
            scheduler.step()

        # epoch mean
        epoch_mean = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        loss_epoch_mean.append(epoch_mean)

        # -----------------------------
        # Save checkpoint every epoch (like your current checkpoint_last.pt)
        # -----------------------------
        checkpoint: Dict[str, Any] = {
            "phase":           "train_on_digits",
            "digits":          digits,
            "epoch":           ep,
            "model_state_dict": ddpm.state_dict(),
            "n_T":             n_T,
            "n_classes":       n_classes,
            "n_feat":          n_feat,
            "loss_epoch_mean": loss_epoch_mean,
            "update_frequency": update_frequency,
        }

        if use_frequency_updates:
            checkpoint.update(
                {
                    "optimizer_state_dicts": {
                        "shallow": opt_shallow.state_dict(),
                        "middle":  opt_middle.state_dict(),
                        "deep":    opt_deep.state_dict(),
                    },
                    "scheduler_state_dicts": {
                        "shallow": sch_shallow.state_dict(),
                        "middle":  sch_middle.state_dict(),
                        "deep":    sch_deep.state_dict(),
                    },
                    "depth_groups": {
                        "shallow_modules": ["init_conv", "up2", "out"],
                        "middle_modules":  ["down1", "up1", "timeembed2", "contextembed2"],
                        "deep_modules":    ["down2", "to_vec", "up0", "timeembed1", "contextembed1"],
                    },
                }
            )
        else:
            checkpoint.update(
                {
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
            )
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
    early_stop_loss:      float = 0.025,
    n_epoch:              int = 100,
    batch_size:           int = 256,
    n_T:                  int = 400,
    n_classes:            int = 10,
    n_feat:               int = 128,
    lrate:                float = 1e-4,
    num_workers:          int = 5,
    # Nested-Learning style update schedule
    # update_frequency = [shallow, middle, deep] (e.g. [1,4,16])
    update_frequency: Optional[List[int]] = None,
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
        digits                       =digits_A,
        save_dir                     =phaseA_dir,
        early_stop_loss              =early_stop_loss,
        n_epoch                      =n_epoch,
        batch_size                   =batch_size,
        n_T                          =n_T,
        n_classes                    =n_classes,
        n_feat                       =n_feat,
        lrate                        =lrate,
        num_workers                  =num_workers,
        update_frequency             =[1,4,16],
        pretrained_path              =None,
        model_out_name               ="model_taskA.pth",
        save_checkpoints_each_epoch  =False,
    )

    # -----------------------------
    # Phase 2: continue on Task B (6–9), starting from model A
    # -----------------------------
    model_A_then_B_path = train_on_digits(
        digits                       =digits_B,
        save_dir                     =phaseB_dir,
        early_stop_loss              =early_stop_loss,
        n_epoch                      =n_epoch,
        batch_size                   =batch_size,
        n_T                          =n_T,
        n_classes                    =n_classes,
        n_feat                       =n_feat,
        lrate                        =lrate,
        num_workers                  =num_workers,
        update_frequency             =[1,4,16],
        pretrained_path              =model_A_path,  # <-- continue from Task A
        model_out_name               ="model_taskA_then_B.pth",
        save_checkpoints_each_epoch  =False,
    )

    print("\n=== DONE ===")
    print("Model after Task A only      :", model_A_path)
    print("Model after Task A then Task B:", model_A_then_B_path)

    return model_A_path, model_A_then_B_path


if __name__ == "__main__":
    # Two-phase continual training (Phase 1: 0–5, Phase 2: 6–9)
    train_two_phase_continual()

"""
Phase 5: Adversarial Training + TRADES

Provides:
  - adversarial_train()   — fine-tune MLP/CNN1D with mixed surrogate-based AT
  - trades_train()        — TRADES loss adversarial training (self-generated adv)
  - smoothing_predict()   — input smoothing defense (inference-time)
  - fgsm_batch()          — inline FGSM for AT loop
  - pgd_batch()           — inline PGD for AT loop (10 steps, fast)
  - pgd_eval_batch()      — inline PGD for eval (40 steps, full)
  - hsja_subset()         — simplified HopSkipJump for decision-based eval

All GPU (cuda). Surrogate is ALWAYS frozen — its weights are never updated.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("phase5.adv_train")


# ---------------------------------------------------------------------------
# Inline attack helpers (operate on GPU tensors, no numpy)
# ---------------------------------------------------------------------------

def fgsm_batch(
    surrogate: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    eps_lo: Optional[torch.Tensor] = None,
    eps_hi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FGSM using surrogate gradients. Returns adversarial x on same device."""
    x_adv = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(surrogate(x_adv), y)
    surrogate.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.detach().sign()

    if eps_lo is not None and eps_hi is not None:
        # Per-feature epsilon bounds
        delta = torch.clamp(eps * grad_sign, eps_lo, eps_hi)
    else:
        delta = eps * grad_sign

    return torch.clamp(x + delta, 0.0, 1.0).detach()


def pgd_batch(
    surrogate: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step_size: float,
    n_steps: int,
    eps_lo: Optional[torch.Tensor] = None,
    eps_hi: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """PGD using surrogate gradients. Returns adversarial x on same device."""
    if seed is not None:
        torch.manual_seed(seed)

    if eps_lo is not None and eps_hi is not None:
        # Feature-aware random start: uniform within per-feature bounds
        delta = torch.stack([
            torch.empty(x.shape[0]).uniform_(float(eps_lo[j]), max(float(eps_hi[j]), float(eps_lo[j]) + 1e-9))
            for j in range(x.shape[1])
        ], dim=1).to(x.device)
    else:
        delta = torch.empty_like(x).uniform_(-eps, eps)

    x_adv = torch.clamp(x + delta, 0.0, 1.0)

    for _ in range(n_steps):
        x_adv = x_adv.requires_grad_(True)
        loss = F.cross_entropy(surrogate(x_adv), y)
        surrogate.zero_grad()
        loss.backward()
        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv.detach() + step_size * grad_sign

        if eps_lo is not None and eps_hi is not None:
            delta = torch.clamp(x_adv - x, eps_lo, eps_hi)
        else:
            delta = torch.clamp(x_adv - x, -eps, eps)

        x_adv = torch.clamp(x + delta, 0.0, 1.0)

    return x_adv.detach()


def _apply_aware_batch(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    eps_lo: Optional[torch.Tensor],
    eps_hi: Optional[torch.Tensor],
) -> torch.Tensor:
    """Re-enforce feature-aware delta bounds and [0,1] clip."""
    if eps_lo is None or eps_hi is None:
        return torch.clamp(x_adv, 0.0, 1.0)
    delta = torch.clamp(x_adv - x_orig, eps_lo, eps_hi)
    return torch.clamp(x_orig + delta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Adversarial training (surrogate-based)
# ---------------------------------------------------------------------------

def adversarial_train(
    model: nn.Module,
    surrogate: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    save_dir,
    model_name: str,
    eps_lo: Optional[torch.Tensor] = None,
    eps_hi: Optional[torch.Tensor] = None,
) -> Tuple[List[Dict], str]:
    """
    Fine-tune `model` with surrogate-based adversarial training.
    Surrogate weights are NEVER updated.

    Args:
        model       : Pre-trained model to harden (moved to device by caller).
        surrogate   : Frozen surrogate for attack generation.
        train_loader: Clean training data loader.
        val_loader  : Validation data loader (for robust early stopping).
        cfg         : adversarial_training section of config.
        eps_lo/hi   : Per-feature epsilon bounds (D,) tensors on device, or None.
        save_dir    : Path to save checkpoints.
        model_name  : "mlp" or "cnn1d".

    Returns:
        (training_curves, best_ckpt_path)
    """
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Attack mixture: normalise weights
    attack_cfgs = cfg["training_attacks"]
    weights = np.array([a["weight"] for a in attack_cfgs], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(42)

    pgd_steps = cfg.get("pgd_train_steps", 10)
    pgd_ss    = cfg.get("learning_rate", 0.0005)   # reuse or hardcode
    pgd_ss    = 0.005   # standard: eps/n_steps ~= 0.03/6

    best_robust = 0.0
    patience = 0
    training_curves = []
    best_ckpt = str(save_dir / f"{model_name}_hardened_best.pt")
    epoch_ckpt_fmt = str(save_dir / f"{model_name}_epoch_{{:02d}}.pt")

    logger.info("[%s] Starting AT fine-tune: %d epochs, batch=%d, lr=%.5f, amp=%s",
                model_name, cfg["epochs"], cfg["batch_size"], cfg["learning_rate"], use_amp)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            n = x_batch.shape[0]
            n_clean = n // 2
            x_clean_b = x_batch[:n_clean]
            y_clean_b = y_batch[:n_clean]
            x_adv_b   = x_batch[n_clean:]
            y_adv_b   = y_batch[n_clean:]

            # Sample one attack config for this batch
            atk_idx = int(rng.choice(len(attack_cfgs), p=weights))
            atk = attack_cfgs[atk_idx]
            eps = float(atk["eps"])

            with torch.enable_grad():
                # Generate adversarial from surrogate (surrogate is frozen)
                if atk["attack"] == "fgsm":
                    x_adv_gen = fgsm_batch(surrogate, x_adv_b, y_adv_b, eps, eps_lo, eps_hi)
                else:
                    x_adv_gen = pgd_batch(surrogate, x_adv_b, y_adv_b, eps,
                                          pgd_ss, pgd_steps, eps_lo, eps_hi)

            x_combined = torch.cat([x_clean_b, x_adv_gen], dim=0)
            y_combined = torch.cat([y_clean_b, y_adv_b],   dim=0)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_combined)
                loss   = F.cross_entropy(logits, y_combined)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item() * n
            total_correct += (logits.detach().argmax(1) == y_combined).sum().item()
            total_n       += n

        scheduler.step()
        train_loss = total_loss / max(1, total_n)
        train_acc  = total_correct / max(1, total_n)

        # Validation: clean acc + PGD robust acc
        clean_acc, robust_fgsm, robust_pgd = _validate_robust(
            model, surrogate, val_loader, device,
            eps=0.03, pgd_steps=10, eps_lo=eps_lo, eps_hi=eps_hi, use_amp=use_amp
        )

        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  4),
            "val_clean_acc":   round(clean_acc,   4),
            "val_robust_fgsm": round(robust_fgsm, 4),
            "val_robust_pgd":  round(robust_pgd,  4),
            "epoch_time_s":    round(epoch_time, 1),
        }
        training_curves.append(row)
        logger.info(
            "[%s][Epoch %02d/%d] loss=%.4f train=%.4f clean=%.4f fgsm=%.4f pgd=%.4f (%.1fs)",
            model_name, epoch, cfg["epochs"],
            train_loss, train_acc, clean_acc, robust_fgsm, robust_pgd, epoch_time
        )

        # Save epoch checkpoint (for crash recovery)
        torch.save(model.state_dict(), epoch_ckpt_fmt.format(epoch))

        # Early stopping on PGD robust val acc
        if robust_pgd > best_robust + 1e-4:
            best_robust = robust_pgd
            patience = 0
            torch.save(model.state_dict(), best_ckpt)
            logger.info("  -> New best robust val_acc=%.4f, saved checkpoint", best_robust)
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                logger.info("  -> Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch, patience)
                break

    logger.info("[%s] AT fine-tune done. Best val robust PGD acc=%.4f", model_name, best_robust)
    return training_curves, best_ckpt


def _validate_robust(
    model: nn.Module,
    surrogate: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    eps: float,
    pgd_steps: int,
    eps_lo: Optional[torch.Tensor],
    eps_hi: Optional[torch.Tensor],
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Return (clean_acc, fgsm_robust_acc, pgd_robust_acc) on val_loader."""
    model.eval()
    surrogate.eval()
    pgd_ss = eps / max(pgd_steps // 2, 1)

    clean_correct = fgsm_correct = pgd_correct = total = 0

    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        n = x_batch.shape[0]

        # Clean
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds_clean = model(x_batch).argmax(1)
        clean_correct += (preds_clean == y_batch).sum().item()

        # FGSM adversarial
        x_fgsm = fgsm_batch(surrogate, x_batch, y_batch, eps, eps_lo, eps_hi)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds_fgsm = model(x_fgsm).argmax(1)
        fgsm_correct += (preds_fgsm == y_batch).sum().item()

        # PGD adversarial (fewer steps for speed during training)
        x_pgd = pgd_batch(surrogate, x_batch, y_batch, eps, pgd_ss, pgd_steps, eps_lo, eps_hi)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds_pgd = model(x_pgd).argmax(1)
        pgd_correct += (preds_pgd == y_batch).sum().item()

        total += n

    return (clean_correct / total, fgsm_correct / total, pgd_correct / total)


# ---------------------------------------------------------------------------
# TRADES training
# ---------------------------------------------------------------------------

def trades_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg_trades: dict,
    device: torch.device,
    save_dir,
    model_name: str,
    eps_lo: Optional[torch.Tensor] = None,
    eps_hi: Optional[torch.Tensor] = None,
) -> Tuple[List[Dict], str]:
    """
    TRADES (Zhang et al. 2019) adversarial training using the MODEL's own gradients.
    loss = CE(clean) + beta * KL(clean_softmax || adv_softmax)
    Adversarial examples are generated by maximising KL divergence from the MODEL,
    NOT from the surrogate. This is a separate defense paradigm.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_trades["learning_rate"],
        weight_decay=cfg_trades["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_trades["epochs"], eta_min=1e-6
    )

    use_amp = cfg_trades.get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    eps   = float(cfg_trades["epsilon"])
    ss    = float(cfg_trades["step_size"])
    steps = int(cfg_trades["perturb_steps"])
    beta  = float(cfg_trades["beta"])

    best_clean = 0.0
    best_ckpt = str(save_dir / f"{model_name}_trades_best.pt")
    epoch_ckpt_fmt = str(save_dir / f"{model_name}_trades_epoch_{{:02d}}.pt")

    patience = 0
    training_curves = []

    logger.info("[%s-TRADES] epochs=%d, eps=%.3f, beta=%.1f, steps=%d",
                model_name, cfg_trades["epochs"], eps, beta, steps)

    for epoch in range(1, cfg_trades["epochs"] + 1):
        model.train()
        t0 = time.time()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Generate TRADES adversarial: maximise KL(natural || adv) from MODEL
            model.eval()
            with torch.no_grad():
                logits_nat = model(x_batch)
                probs_nat  = F.softmax(logits_nat.detach(), dim=1)

            delta = torch.empty_like(x_batch).uniform_(-eps, eps)
            x_adv = torch.clamp(x_batch + delta, 0.0, 1.0)

            for _ in range(steps):
                x_adv = x_adv.requires_grad_(True)
                logits_adv = model(x_adv)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_adv, dim=1),
                    probs_nat,
                    reduction="batchmean",
                )
                grad = torch.autograd.grad(loss_kl, x_adv)[0].detach().sign()
                x_adv = x_adv.detach() + ss * grad
                if eps_lo is not None and eps_hi is not None:
                    delta = torch.clamp(x_adv - x_batch, eps_lo, eps_hi)
                else:
                    delta = torch.clamp(x_adv - x_batch, -eps, eps)
                x_adv = torch.clamp(x_batch + delta, 0.0, 1.0)

            x_adv = x_adv.detach()
            model.train()

            # TRADES loss
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_nat2 = model(x_batch)
                logits_adv2 = model(x_adv)
                loss_nat    = F.cross_entropy(logits_nat2, y_batch)
                loss_rob    = F.kl_div(
                    F.log_softmax(logits_adv2, dim=1),
                    F.softmax(logits_nat2.detach(), dim=1),
                    reduction="batchmean",
                )
                loss = loss_nat + beta * loss_rob

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item() * x_batch.shape[0]
            total_correct += (logits_nat2.detach().argmax(1) == y_batch).sum().item()
            total_n       += x_batch.shape[0]

        scheduler.step()
        train_loss = total_loss / max(1, total_n)
        train_acc  = total_correct / max(1, total_n)

        # Val: clean acc only (TRADES typically monitors clean acc)
        clean_acc, _, _ = _validate_robust(
            model, model, val_loader, device,  # pass model as "surrogate" for dummy fgsm
            eps=eps, pgd_steps=3, eps_lo=eps_lo, eps_hi=eps_hi, use_amp=use_amp
        )

        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 4),
            "val_clean_acc": round(clean_acc, 4),
            "epoch_time_s": round(epoch_time, 1),
        }
        training_curves.append(row)
        logger.info(
            "[%s-TRADES][Epoch %02d/%d] loss=%.4f train=%.4f val_clean=%.4f (%.1fs)",
            model_name, epoch, cfg_trades["epochs"],
            train_loss, train_acc, clean_acc, epoch_time
        )

        torch.save(model.state_dict(), epoch_ckpt_fmt.format(epoch))

        if clean_acc > best_clean + 1e-4:
            best_clean = clean_acc
            patience = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience += 1
            if patience >= cfg_trades["early_stopping_patience"]:
                logger.info("  -> TRADES early stopping at epoch %d", epoch)
                break

    return training_curves, best_ckpt


# ---------------------------------------------------------------------------
# Input smoothing defense
# ---------------------------------------------------------------------------

def smoothing_predict(
    model: nn.Module,
    x: torch.Tensor,
    noise_std: float,
    n_samples: int,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Randomised smoothing: average softmax predictions over n_samples noisy copies.
    Returns aggregated logits (not yet argmax).
    """
    model.eval()
    agg = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    vote_counts = torch.zeros(x.shape[0], model.fc.out_features
                              if hasattr(model, 'fc') else
                              model.net[-1].out_features, device=device)

    with torch.no_grad():
        for _ in range(n_samples):
            x_noisy = torch.clamp(x + torch.randn_like(x) * noise_std, 0.0, 1.0)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_noisy)
            vote_counts += F.softmax(logits.float(), dim=1)

    return vote_counts  # caller does argmax


# ---------------------------------------------------------------------------
# Simplified HSJA (HopSkipJump) for decision-based evaluation
# ---------------------------------------------------------------------------

def hsja_batch(
    target_model: nn.Module,
    x_clean: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float,
    n_iterations: int,
    binary_search_steps: int,
    device: torch.device,
    eps_lo: Optional[torch.Tensor] = None,
    eps_hi: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Simplified HopSkipJump Attack (decision-based, hard-label only).

    Per sample:
      1. Find initial adversarial via random perturbations.
      2. Binary search on clean-to-adversarial line to find boundary.
      3. Estimate gradient at boundary via random sign estimation (Monte Carlo).
      4. Step along estimated gradient; maintain adversarial status.
      5. Final projection to L∞ epsilon ball.

    Uses only hard-label (predict) from target_model — NO gradients from target.

    Args:
        target_model     : Hardened target model (what we're attacking).
        x_clean          : (N, D) clean samples.
        y_true           : (N,) true labels.
        epsilon          : L∞ budget for final clip.
        n_iterations     : Gradient-estimation + step iterations.
        binary_search_steps: Binary search depth at each iteration.

    Returns:
        X_adv : (N, D) adversarial samples (or clean if no adversarial found).
    """
    torch.manual_seed(seed)
    target_model.eval()
    N, D = x_clean.shape
    X_adv = x_clean.clone()

    # n_est: samples for gradient estimation
    n_est = max(50, D)

    for i in range(N):
        x = x_clean[i:i+1]  # (1, D)
        y = y_true[i].item()

        # ---- Phase 1: Find initial adversarial ----
        x_adv_init = None
        for _ in range(50):
            noise = torch.randn_like(x)
            noise = noise / (noise.norm() + 1e-12) * epsilon * 5.0
            cand = torch.clamp(x + noise, 0.0, 1.0)
            with torch.no_grad():
                pred = target_model(cand).argmax(1).item()
            if pred != y:
                x_adv_init = cand.clone()
                break

        if x_adv_init is None:
            # Try fully random
            for _ in range(20):
                cand = torch.rand_like(x)
                with torch.no_grad():
                    pred = target_model(cand).argmax(1).item()
                if pred != y:
                    x_adv_init = cand.clone()
                    break

        if x_adv_init is None:
            continue  # Give up; leave as clean

        x_adv = x_adv_init.clone()

        # ---- Phase 2: Iterative boundary walk ----
        for _ in range(n_iterations):
            # 2a. Binary search on line from x_adv to x
            lo, hi = 0.0, 1.0
            for _ in range(binary_search_steps):
                mid = (lo + hi) / 2.0
                x_mid = (1 - mid) * x_adv + mid * x
                x_mid = torch.clamp(x_mid, 0.0, 1.0)
                with torch.no_grad():
                    pred_mid = target_model(x_mid).argmax(1).item()
                if pred_mid != y:
                    lo = mid  # x_mid is still adversarial; move closer to x
                    x_adv = x_mid
                else:
                    hi = mid  # x_mid is clean; boundary is further from x

            # 2b. Estimate gradient direction at boundary via random sign estimation
            n_correct = 0
            grad_est  = torch.zeros_like(x)
            batch_noise = torch.randn(n_est, D, device=device)
            batch_noise = batch_noise / (batch_noise.norm(dim=1, keepdim=True) + 1e-12) * 0.01

            cands = torch.clamp(x_adv + batch_noise, 0.0, 1.0)
            with torch.no_grad():
                preds_est = target_model(cands).argmax(1)
            adv_mask = (preds_est != y).float()  # 1 if adversarial

            # Gradient estimate: mean of perturbations that stayed adversarial
            grad_est = (batch_noise * adv_mask.unsqueeze(1)).mean(dim=0, keepdim=True)

            if grad_est.norm() > 1e-9:
                # 2c. Step along gradient estimate
                step_size = epsilon * 0.1
                x_cand = torch.clamp(x_adv + step_size * grad_est.sign(), 0.0, 1.0)
                with torch.no_grad():
                    pred_cand = target_model(x_cand).argmax(1).item()
                if pred_cand != y:
                    x_adv = x_cand

        # Project to epsilon ball and clip
        if eps_lo is not None and eps_hi is not None:
            delta = torch.clamp(x_adv - x, eps_lo.unsqueeze(0), eps_hi.unsqueeze(0))
        else:
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        X_adv[i] = x_adv.squeeze(0)

    return X_adv.detach()

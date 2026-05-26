"""
04_train_diffusion_models.py
-----------------------------
Phase 4: Train exactly 4 conditional DDPM models
(1 schedule type: linear × 4 diffusion step counts: 400, 800, 1200, 2000).

All models use identical fixed architecture and 200-epoch training.
No validation GMV monitoring for early stopping.

Outputs (per model)
-------------------
artifacts/models/ddpm_schedule-{schedule_type}_T-{T}_seed-42.pt
artifacts/training_logs/ddpm_schedule-{schedule_type}_T-{T}_seed-42.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.datasets import load_dataset
from src.train import train_one_conditional_ddpm
from src.utils import get_logger, set_global_seed

logger = get_logger("04_train_diffusion_models", logging.INFO)


def main() -> None:
    cfg = get_config()

    processed_dir = Path("data/processed")
    model_dir = Path("artifacts/models")
    log_dir = Path("artifacts/training_logs")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.random_seed)

    logger.info("=" * 60)
    logger.info("STEP 4 – Train 4 conditional DDPM models (linear schedule, T ∈ {400, 800, 1200, 2000})")
    logger.info("=" * 60)

    # ---- Load training data ----------------------------------------------
    logger.info("Loading training dataset …")
    train_ds = load_dataset(processed_dir / "covariance_pairs_train.npz")
    logger.info(
        "Training pairs: %d, cond shape %s",
        len(train_ds["condition_scaled"]), train_ds["condition_scaled"].shape,
    )

    # ---- Grid of model configs to train -----------------------------------
    schedule_grid = cfg.training["beta_schedule_grid"]
    T_grid = cfg.training["diffusion_steps_grid"]

    total = len(schedule_grid) * len(T_grid)
    counter = 0

    for schedule_type in schedule_grid:
        for T in T_grid:
            counter += 1
            logger.info(
                "\n[%d/%d] Training: schedule=%s, T=%d",
                counter, total, schedule_type, T,
            )

            # Check if checkpoint already exists
            seed = cfg.random_seed
            ckpt_name = f"ddpm_schedule-{schedule_type}_T-{T}_seed-{seed}.pt"
            ckpt_path = model_dir / ckpt_name
            if ckpt_path.exists():
                logger.info("  Checkpoint already exists: %s – SKIPPING.", ckpt_path)
                continue

            set_global_seed(seed)

            model, history = train_one_conditional_ddpm(
                train_dataset=train_ds,
                schedule_type=schedule_type,
                T=T,
                beta_min=cfg.training["beta_min"],
                beta_max=cfg.training["beta_max"],
                hidden_dim=cfg.model["hidden_dim"],
                num_hidden=cfg.model["num_hidden_layers"],
                time_embed_dim=cfg.model["time_embedding_dim"],
                dropout=cfg.model["dropout"],
                epochs=cfg.training["epochs"],
                batch_size=cfg.training["batch_size"],
                learning_rate=cfg.training["learning_rate"],
                weight_decay=cfg.training["weight_decay"],
                seed=seed,
                save_dir=None,
            )

            import torch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "schedule_type": schedule_type,
                    "T": T,
                    "seed": seed,
                    "hidden_dim": cfg.model["hidden_dim"],
                    "num_hidden": cfg.model["num_hidden_layers"],
                    "time_embed_dim": cfg.model["time_embedding_dim"],
                    "dropout": cfg.model["dropout"],
                    "epochs_trained": cfg.training["epochs"],
                },
                ckpt_path,
            )
            log_path = log_dir / f"ddpm_schedule-{schedule_type}_T-{T}_seed-{seed}.csv"
            history.to_csv(log_path, index=False)
            logger.info("  Saved checkpoint: %s", ckpt_path)

    logger.info("\nStep 4 complete – trained %d model(s).", counter)


if __name__ == "__main__":
    main()

"""
Production training loop for SAGE Qwen2.5-VL with 3-level tqdm progress bars,
hard 10-hour compute budget enforcement, step-wise resumable checkpointing,
and Macro-F1 driven best model selection.
"""

import os
import sys
import time
import shutil
import random
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from src.utils import BudgetManager, save_json, load_json, ensure_dirs
from src.dataset import PipelineProfiler
from src.evaluation import evaluate_model

logger = logging.getLogger("SAGE.trainer")


class SAGETrainer:
    """
    Kaggle-optimized, resilient trainer for Qwen2.5-VL with LoRA.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        resume_checkpoint_dir: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.resume_dir = resume_checkpoint_dir

        self.train_cfg = config.get("training", {})
        self.eval_cfg = config.get("evaluation", {})
        self.checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "checkpoints")
        self.artifacts_dir = config.get("paths", {}).get("artifacts_dir", "artifacts")

        ensure_dirs(
            self.checkpoints_dir,
            os.path.join(self.checkpoints_dir, "latest"),
            os.path.join(self.checkpoints_dir, "best"),
            self.artifacts_dir,
        )

        # Hyperparameters
        self.num_epochs = self.train_cfg.get("epochs", 2)
        self.batch_size = self.train_cfg.get("batch_size", 8)
        self.grad_accum_steps = self.train_cfg.get("gradient_accumulation_steps", 2)
        self.learning_rate = float(self.train_cfg.get("learning_rate", 2e-4))
        self.weight_decay = float(self.train_cfg.get("weight_decay", 0.01))
        self.warmup_ratio = float(self.train_cfg.get("warmup_ratio", 0.03))
        self.max_grad_norm = float(self.train_cfg.get("max_grad_norm", 1.0))
        self.logging_steps = int(self.train_cfg.get("logging_steps", 10))
        self.eval_steps = int(self.train_cfg.get("eval_steps", 3000))
        self.save_steps = int(self.train_cfg.get("save_steps", 3000))
        self.max_training_hours = float(self.train_cfg.get("max_training_hours", 9.0))

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler steps calculation
        self.steps_per_epoch = len(self.train_dataloader) // self.grad_accum_steps
        self.total_training_steps = self.steps_per_epoch * self.num_epochs
        warmup_steps = int(self.total_training_steps * self.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_training_steps,
        )

        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -1.0
        self.best_step = 0
        self.best_epoch = 0
        self.training_history = []
        self.start_time = time.time()
        self.budget_manager = BudgetManager(max_hours=self.max_training_hours, start_time=self.start_time)
        self.profiler = PipelineProfiler()

        # Handle checkpoint resuming
        if self.resume_dir and os.path.exists(self.resume_dir):
            self._load_checkpoint(self.resume_dir)

    def _save_checkpoint(self, checkpoint_name: str, val_metric: Optional[float] = None) -> str:
        """
        Saves full training state for 100% resumable training:
        - LoRA adapter weights
        - optimizer state
        - scheduler state
        - RNG states
        - trainer_state.json
        """
        dest_dir = os.path.join(self.checkpoints_dir, checkpoint_name)
        ensure_dirs(dest_dir)

        # Save LoRA adapter
        self.model.save_pretrained(dest_dir)

        # Save optimizer & scheduler
        torch.save(self.optimizer.state_dict(), os.path.join(dest_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(dest_dir, "scheduler.pt"))

        # Save RNG states
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, os.path.join(dest_dir, "rng_state.pt"))

        # Save Trainer state
        trainer_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "best_epoch": self.best_epoch,
            "val_metric": val_metric,
            "elapsed_seconds": self.budget_manager.elapsed_seconds(),
            "elapsed_str": self.budget_manager.format_time(self.budget_manager.elapsed_seconds()),
            "profiler": self.profiler.summary(),
        }
        save_json(trainer_state, os.path.join(dest_dir, "trainer_state.json"))

        # Save copy of config for reproducibility
        import yaml
        with open(os.path.join(dest_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        return dest_dir

    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        """Restores complete training state from checkpoint."""
        print(f"\nResuming training from checkpoint: {checkpoint_dir}")
        state_file = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(state_file):
            state = load_json(state_file)
            self.current_epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            self.best_metric = state.get("best_metric", -1.0)
            self.best_step = state.get("best_step", 0)
            self.best_epoch = state.get("best_epoch", 0)
            resumed_elapsed = state.get("elapsed_seconds", 0.0)
            # Adjust start time so remaining budget is properly computed
            self.start_time = time.time() - resumed_elapsed
            self.budget_manager = BudgetManager(max_hours=self.max_training_hours, start_time=self.start_time)
            print(f"Restored: Epoch {self.current_epoch}, Step {self.global_step}, Best Metric: {self.best_metric}")

        opt_file = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_file):
            self.optimizer.load_state_dict(torch.load(opt_file, map_location=self.device))
            print("Restored optimizer state.")

        sched_file = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(sched_file):
            self.scheduler.load_state_dict(torch.load(sched_file))
            print("Restored learning rate scheduler.")

        rng_file = os.path.join(checkpoint_dir, "rng_state.pt")
        if os.path.exists(rng_file):
            rng = torch.load(rng_file)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            print("Restored deterministic random states.")

    def run_validation(self, is_final: bool = False) -> Dict[str, Any]:
        """Runs validation evaluation and logs diagnostic results."""
        max_samples = None if is_final else self.eval_cfg.get("eval_subset_size", 1000)
        metrics, _ = evaluate_model(
            model=self.model,
            processor=self.processor,
            dataloader=self.val_dataloader,
            device=self.device,
            max_samples=max_samples,
            max_new_tokens=self.eval_cfg.get("generation_max_new_tokens", 128),
            metrics_dir=os.path.join(self.artifacts_dir, "metrics"),
            failures_dir=os.path.join(self.artifacts_dir, "failures"),
        )
        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Executes complete training pipeline with 3-level tqdm bars and budget checks.
        """
        total_samples = len(self.train_dataloader.dataset) * self.num_epochs
        total_batches = len(self.train_dataloader)

        print("\n" + "=" * 65)
        print(" STARTING SAGE PRODUCTION TRAINING")
        print("=" * 65)
        print(f" Total Epochs:               {self.num_epochs}")
        print(f" Train Samples per Epoch:   {len(self.train_dataloader.dataset):,}")
        print(f" Total Training Steps:      {self.total_training_steps:,}")
        print(f" Max Budget Limit:          {self.max_training_hours} Hours")
        print(f" Gradient Accumulation:     {self.grad_accum_steps}")
        print(f" Effective Batch Size:      {self.batch_size * self.grad_accum_steps}")
        print("=" * 65 + "\n")

        # LEVEL 1: TOTAL TRAINING PROGRESS BAR
        total_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_step,
            desc="TOTAL TRAINING",
            position=0,
            leave=True,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        budget_exceeded = False
        latest_loss = 0.0

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.model.train()
            self.optimizer.zero_grad()

            epoch_loss = 0.0
            accumulated_loss = 0.0

            # LEVEL 2: EPOCH PROGRESS BAR
            epoch_desc = f"Epoch {epoch + 1}/{self.num_epochs}"
            epoch_bar = tqdm(
                total=total_batches,
                desc=epoch_desc,
                position=1,
                leave=False,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            )

            # LEVEL 3: BATCH PROGRESS BAR
            batch_bar = tqdm(
                total=self.grad_accum_steps,
                desc="Batch Accumulation",
                position=2,
                leave=False,
                bar_format="{desc} {percentage:3.0f}%|{bar}| [{elapsed}] {postfix}",
            )

            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(self.train_dataloader):
                self.profiler.mark_data_ready()

                # Check budget constraint before executing step
                if self.budget_manager.is_budget_exceeded():
                    budget_exceeded = True
                    print(f"\n[BUDGET] 10-Hour compute budget ({self.max_training_hours}h) reached. Initiating clean exit.")
                    break

                # Prepare model inputs
                model_inputs = {}
                for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"):
                    if k in batch and isinstance(batch[k], torch.Tensor):
                        model_inputs[k] = batch[k].to(self.device)

                # Forward pass
                outputs = self.model(**model_inputs)
                loss = outputs.loss / self.grad_accum_steps
                loss.backward()

                accumulated_loss += loss.item() * self.grad_accum_steps
                batch_bar.update(1)
                batch_bar.set_postfix({"loss": f"{outputs.loss.item():.4f}", "step": f"{(batch_idx % self.grad_accum_steps) + 1}/{self.grad_accum_steps}"})

                # Optimizer step upon completing gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    latest_loss = accumulated_loss / self.grad_accum_steps
                    epoch_loss += accumulated_loss
                    accumulated_loss = 0.0

                    # Update Level 1 & Level 2 bars
                    total_bar.update(1)
                    curr_lr = self.scheduler.get_last_lr()[0]
                    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                    samples_processed = (self.global_step * self.batch_size * self.grad_accum_steps)

                    status = self.budget_manager.get_progress_status(self.global_step, self.total_training_steps)
                    total_bar.set_postfix({
                        "Epoch": f"{epoch + 1}/{self.num_epochs}",
                        "Elapsed": status["elapsed_str"],
                        "ETA": status["eta_str"],
                        "Samples": f"{samples_processed:,}",
                    })

                    epoch_bar.set_postfix({
                        "loss": f"{latest_loss:.4f}",
                        "lr": f"{curr_lr:.2e}",
                        "VRAM": f"{vram_gb:.1f}GB",
                        "wait%": f"{self.profiler.get_wait_percentage()}%",
                    })

                    batch_bar.reset()

                    # Periodic step evaluation
                    if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                        val_metrics = self.run_validation(is_final=False)
                        val_f1 = val_metrics.get("macro_f1", 0.0)
                        print(f"\n[Step {self.global_step}] Validation Macro F1: {val_f1:.4f} | Accuracy: {val_metrics.get('normalized_accuracy', 0.0):.4f}")

                        # Save latest checkpoint
                        self._save_checkpoint("latest", val_metric=val_f1)

                        # Save best checkpoint based primarily on Macro F1
                        if val_f1 > self.best_metric:
                            self.best_metric = val_f1
                            self.best_step = self.global_step
                            self.best_epoch = epoch + 1
                            self._save_checkpoint("best", val_metric=val_f1)
                            print(f"[BEST] New best checkpoint saved with Macro F1: {val_f1:.4f}")

                        self.model.train()

                    # Periodic step checkpoint
                    elif self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint("latest", val_metric=None)

                self.profiler.mark_compute_finished()
                epoch_bar.update(1)

            epoch_bar.close()
            batch_bar.close()

            # End of epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1:02d}", val_metric=None)
            self._save_checkpoint("latest", val_metric=None)

            # End of epoch validation
            if self.eval_cfg.get("full_eval_at_epoch_end", True) and not budget_exceeded:
                val_metrics = self.run_validation(is_final=False)
                val_f1 = val_metrics.get("macro_f1", 0.0)
                print(f"\n[Epoch {epoch + 1}] Validation Macro F1: {val_f1:.4f} | Accuracy: {val_metrics.get('normalized_accuracy', 0.0):.4f}")
                if val_f1 > self.best_metric:
                    self.best_metric = val_f1
                    self.best_step = self.global_step
                    self.best_epoch = epoch + 1
                    self._save_checkpoint("best", val_metric=val_f1)
                    print(f"[BEST] New best checkpoint saved with Macro F1: {val_f1:.4f}")

            if budget_exceeded:
                break

        total_bar.close()

        # Clean budget exit: ensure latest checkpoint is intact
        self._save_checkpoint("latest", val_metric=self.best_metric)

        elapsed_sec = self.budget_manager.elapsed_seconds()
        train_stats = {
            "total_train_samples": len(self.train_dataloader.dataset),
            "total_val_samples": len(self.val_dataloader.dataset),
            "total_steps": self.global_step,
            "epochs_completed": self.current_epoch + 1,
            "elapsed_seconds": round(elapsed_sec, 2),
            "elapsed_time_str": self.budget_manager.format_time(elapsed_sec),
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "best_macro_f1": round(self.best_metric, 4),
            "profiler": self.profiler.summary(),
        }

        # Save training summary JSON
        save_json(train_stats, os.path.join(self.artifacts_dir, "training_summary.json"))
        return train_stats

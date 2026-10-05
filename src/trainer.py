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

from src.utils import BudgetManager, save_json, load_json, ensure_dirs, get_device_info, resolve_dtype
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

        # Precision & device context for AMP autocast (prevents FP16 softmax overflow)
        dev_info = get_device_info()
        self.compute_dtype = resolve_dtype(
            config.get("model", {}).get("dtype", "auto"),
            dev_info.get("bf16_supported", False),
        )
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            foreach=False,
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
            saved_epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            self.best_metric = state.get("best_metric", -1.0)
            self.best_step = state.get("best_step", 0)
            self.best_epoch = state.get("best_epoch", 0)
            resumed_elapsed = state.get("elapsed_seconds", 0.0)

            # Advance epoch if the checkpoint was saved at end-of-epoch
            if self.steps_per_epoch > 0 and self.global_step >= (saved_epoch + 1) * self.steps_per_epoch:
                self.current_epoch = saved_epoch + 1
                print(f"Saved checkpoint was at completion of Epoch {saved_epoch + 1}. Resuming at Epoch {self.current_epoch + 1}.")
            else:
                self.current_epoch = saved_epoch

            # Adjust start time so remaining budget is properly computed
            self.start_time = time.time() - resumed_elapsed
            self.budget_manager = BudgetManager(max_hours=self.max_training_hours, start_time=self.start_time)
            print(f"Restored: Epoch {self.current_epoch + 1}/{self.num_epochs}, Step {self.global_step}/{self.total_training_steps}, Best Metric: {self.best_metric}")

        opt_file = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_file):
            try:
                self.optimizer.load_state_dict(torch.load(opt_file, map_location=self.device, weights_only=False))
            except TypeError:
                self.optimizer.load_state_dict(torch.load(opt_file, map_location=self.device))
            for p, state in self.optimizer.state.items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=p.device, dtype=p.dtype if k != "step" else v.dtype)
            print("Restored optimizer state.")

        sched_file = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(sched_file):
            # Recalibrate scheduler for total_training_steps to prevent LR dropping to 0
            # when extending training beyond the original checkpoint's step count
            warmup_steps = int(self.total_training_steps * self.warmup_ratio)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_training_steps,
                last_epoch=self.global_step - 1 if self.global_step > 0 else -1,
            )
            curr_lr = self.scheduler.get_last_lr()[0]
            print(f"Recalibrated scheduler for {self.total_training_steps} total steps (current step: {self.global_step}, LR: {curr_lr:.2e}).")

        rng_file = os.path.join(checkpoint_dir, "rng_state.pt")
        if os.path.exists(rng_file):
            try:
                try:
                    rng = torch.load(rng_file, weights_only=False)
                except TypeError:
                    rng = torch.load(rng_file)
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
                print("Restored deterministic random states.")
            except Exception as e:
                logger.warning(f"Could not restore RNG state ({e}), proceeding with current seed.")

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
        Executes complete training pipeline with 2-level tqdm progress bars.

        Progress bars update ONLY on optimizer steps (every grad_accum_steps
        micro-batches), keeping output clean and readable in Kaggle notebooks.
        """
        total_batches = len(self.train_dataloader)

        print("\n" + "=" * 65)
        print(" STARTING SAGE PRODUCTION TRAINING")
        print("=" * 65)
        print(f" Total Epochs:              {self.num_epochs}")
        print(f" Train Samples per Epoch:  {len(self.train_dataloader.dataset):,}")
        print(f" Total Training Steps:     {self.total_training_steps:,}")
        print(f" Max Budget Limit:         {self.max_training_hours} Hours")
        print(f" Gradient Accumulation:    {self.grad_accum_steps}")
        print(f" Effective Batch Size:     {self.batch_size * self.grad_accum_steps}")
        print("=" * 65 + "\n")

        # LEVEL 1: Total training bar — advances once per optimizer step
        total_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_step,
            desc="Training",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

        budget_exceeded = False
        latest_loss = 0.0

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.model.train()
            self.optimizer.zero_grad()

            epoch_loss = 0.0
            accumulated_loss = 0.0

            # LEVEL 2: Epoch bar — counts optimizer steps, NOT raw micro-batches
            epoch_bar = tqdm(
                total=self.steps_per_epoch,
                desc=f"  Epoch {epoch + 1}/{self.num_epochs}",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )

            for batch_idx, batch in enumerate(self.train_dataloader):
                self.profiler.mark_data_ready()

                if self.budget_manager.is_budget_exceeded():
                    budget_exceeded = True
                    tqdm.write(
                        f"\n[BUDGET] Compute budget ({self.max_training_hours}h) reached."
                        " Initiating clean exit."
                    )
                    break

                model_inputs = {}
                for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"):
                    if k in batch and isinstance(batch[k], torch.Tensor):
                        model_inputs[k] = batch[k].to(self.device)

                # AMP Autocast: Keeps matmuls in FP16/BF16 while running Softmax & CrossEntropy in FP32
                with torch.autocast(device_type=self.device_type, dtype=self.compute_dtype):
                    outputs = self.model(**model_inputs)
                    raw_loss = outputs.loss

                # Pre-backward NaN / Inf Guard: Skip corrupted batch without updating gradients
                if raw_loss is None or torch.isnan(raw_loss) or torch.isinf(raw_loss):
                    logger.warning(
                        "[NaN Guard] NaN/Inf loss encountered at step %d, batch %d. Skipping micro-batch.",
                        self.global_step,
                        batch_idx + 1,
                    )
                    self.profiler.mark_compute_finished()
                    continue

                loss = raw_loss / self.grad_accum_steps
                loss.backward()

                accumulated_loss += raw_loss.item()
                self.profiler.mark_compute_finished()

                is_last_batch = (batch_idx + 1) == total_batches
                is_accum_step = (batch_idx + 1) % self.grad_accum_steps == 0

                if is_accum_step or is_last_batch:
                    # Check for NaN / Inf parameter gradients before stepping optimizer
                    has_invalid_grad = False
                    for p in self.model.parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                has_invalid_grad = True
                                break

                    if has_invalid_grad:
                        logger.warning(
                            "[NaN Guard] NaN/Inf gradient detected at step %d. Discarding update to protect weights.",
                            self.global_step,
                        )
                        self.optimizer.zero_grad()
                        accumulated_loss = 0.0
                        continue

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if torch.isnan(torch.as_tensor(grad_norm)) or torch.isinf(torch.as_tensor(grad_norm)):
                        logger.warning(
                            "[NaN Guard] Gradient norm exploded (%.4f) at step %d. Discarding optimizer update.",
                            float(grad_norm),
                            self.global_step,
                        )
                        self.optimizer.zero_grad()
                        accumulated_loss = 0.0
                        continue

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    latest_loss = accumulated_loss / self.grad_accum_steps
                    epoch_loss += accumulated_loss
                    accumulated_loss = 0.0

                    curr_lr = self.scheduler.get_last_lr()[0]
                    vram_gb = (
                        torch.cuda.memory_allocated() / (1024 ** 3)
                        if torch.cuda.is_available() else 0.0
                    )
                    status = self.budget_manager.get_progress_status(
                        self.global_step, self.total_training_steps
                    )

                    # Both bars update ONCE per optimizer step
                    total_bar.set_postfix(
                        loss=f"{latest_loss:.4f}",
                        lr=f"{curr_lr:.2e}",
                        VRAM=f"{vram_gb:.1f}GB",
                        ETA=status["eta_str"],
                        refresh=False,
                    )
                    total_bar.update(1)
                    epoch_bar.set_postfix(
                        loss=f"{latest_loss:.4f}",
                        lr=f"{curr_lr:.2e}",
                        refresh=False,
                    )
                    epoch_bar.update(1)

                    # Periodic structured log line (not per-sample spam)
                    if self.global_step % self.logging_steps == 0:
                        logger.info(
                            "[Step %d/%d] Epoch %d/%d | loss=%.4f | lr=%.2e"
                            " | VRAM=%.1fGB | elapsed=%s | ETA=%s",
                            self.global_step, self.total_training_steps,
                            epoch + 1, self.num_epochs,
                            latest_loss, curr_lr, vram_gb,
                            status["elapsed_str"], status["eta_str"],
                        )

                    # Periodic step evaluation
                    if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                        val_metrics = self.run_validation(is_final=False)
                        val_f1 = val_metrics.get("macro_f1", 0.0)
                        tqdm.write(
                            f"\n[Step {self.global_step}] Val Macro-F1: {val_f1:.4f}"
                            f" | Accuracy: {val_metrics.get("normalized_accuracy", 0.0):.4f}"
                        )
                        self._save_checkpoint("latest", val_metric=val_f1)
                        if val_f1 > self.best_metric:
                            self.best_metric = val_f1
                            self.best_step = self.global_step
                            self.best_epoch = epoch + 1
                            self._save_checkpoint("best", val_metric=val_f1)
                            tqdm.write(f"[BEST] New best checkpoint: Macro-F1={val_f1:.4f}")
                        self.model.train()

                    # Periodic checkpoint without eval
                    elif self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint("latest", val_metric=None)

            epoch_bar.close()

            # End-of-epoch saves
            self._save_checkpoint(f"epoch_{epoch + 1:02d}", val_metric=None)
            self._save_checkpoint("latest", val_metric=None)

            # End-of-epoch validation
            if self.eval_cfg.get("full_eval_at_epoch_end", True) and not budget_exceeded:
                val_metrics = self.run_validation(is_final=False)
                val_f1 = val_metrics.get("macro_f1", 0.0)
                tqdm.write(
                    f"\n[Epoch {epoch + 1}/{self.num_epochs}] Val Macro-F1: {val_f1:.4f}"
                    f" | Accuracy: {val_metrics.get("normalized_accuracy", 0.0):.4f}"
                )
                if val_f1 > self.best_metric:
                    self.best_metric = val_f1
                    self.best_step = self.global_step
                    self.best_epoch = epoch + 1
                    self._save_checkpoint("best", val_metric=val_f1)
                    tqdm.write(f"[BEST] New best checkpoint: Macro-F1={val_f1:.4f}")

            if budget_exceeded:
                break

        total_bar.close()
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
        save_json(train_stats, os.path.join(self.artifacts_dir, "training_summary.json"))
        return train_stats


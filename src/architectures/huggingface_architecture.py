from typing import Dict, Any

import torch
from torch import optim, nn
from torchmetrics import MetricCollection
from torchmetrics.text.rouge import ROUGEScore

from lightning.pytorch import LightningModule

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from transformers import AutoTokenizer


class HuggingFaceArchitecture(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        pretrained_model_name: str,
        strategy: str,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        eta_min_ratio: float,
        interval: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            use_fast=True,
        )
        self.strategy = strategy
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.eta_min_ratio = eta_min_ratio
        self.interval = interval

        metrics = MetricCollection(
            [
                ROUGEScore(),
            ]
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            raise ValueError(f"Invalid model mode: {mode}")
        output = self.model(encoded)
        return output

    def step(
        self,
        batch: Dict[str, Any],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        encoded = batch["encoded"]
        label = encoded["labels"]
        index = batch["index"]
        output = self(
            encoded=encoded,
            mode=mode,
        )
        logit = output.logits
        pred = torch.argmax(
            logit,
            dim=-1,
        )
        loss = output.loss
        if mode == "train":
            return {
                "loss": loss,
                "logit": logit,
                "pred": pred,
                "label": label,
                "index": index,
            }
        else:
            generation = self.model.generate(
                encoded=encoded,
            )
            decoded_generation = self.tokenizer.batch_decode(
                sequences=generation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            decoded_label = self.tokenizer.batch_decode(
                sequences=label,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return {
                "loss": loss,
                "logit": logit,
                "pred": pred,
                "generation": generation,
                "decoded_generation": decoded_generation,
                "label": label,
                "decoded_label": decoded_label,
                "index": index,
            }

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.strategy == "deepspeed_stage_3":
            optimizer = FusedAdam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif (
            self.strategy == "deepspeed_stage_2_offload"
            or self.strategy == "deepspeed_stage_3_offload"
        ):
            optimizer = DeepSpeedCPUAdam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        t_max = total_steps - warmup_steps
        eta_min = self.lr * self.eta_min_ratio

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                main_scheduler,
            ],
            milestones=[
                warmup_steps,
            ],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.interval,
            },
        }

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="train",
        )
        loss = output["loss"]
        pred = output["pred"]
        label = output["label"]
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="eval",
        )
        loss = output["loss"]
        pred = output["pred"]
        generation = output["generation"]
        decoded_generation = output["decoded_generation"]
        label = output["label"]
        decoded_label = output["decoded_label"]
        metrics = self.val_metrics(
            decoded_generation,
            decoded_label,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        return {
            "loss": loss,
            "pred": pred,
            "generation": generation,
            "decoded_generation": decoded_generation,
            "label": label,
            "decoded_label": decoded_label,
        }

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="eval",
        )
        loss = output["loss"]
        pred = output["pred"]
        generation = output["generation"]
        decoded_generation = output["decoded_generation"]
        label = output["label"]
        decoded_label = output["decoded_label"]
        metrics = self.test_metrics(
            decoded_generation,
            decoded_label,
        )
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        return {
            "loss": loss,
            "pred": pred,
            "generation": generation,
            "decoded_generation": decoded_generation,
            "label": label,
            "decoded_label": decoded_label,
        }

    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        encoded = batch["encoded"]
        index = batch["index"]
        index = index.tolist()
        generation = self.model.generate(
            encoded=encoded,
        )
        decoded_generation = self.tokenizer.batch_decode(
            sequences=generation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output = {index[i]: decoded_generation[i] for i in range(len(index))}
        gathered_output = self.all_gather(output)
        return gathered_output

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.test_metrics.reset()

from pathlib import Path
from typing import cast

import torch
from lightning import LightningModule
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from src.module.asr_with_llm import ASRWithLLMModule


class LitASRWithLLMModule(LightningModule):
    def __init__(
        self,
        pretrained_audio_encoder_name_or_path: str | Path,
        pretrained_text_decoder_name_or_path: str | Path,
        learning_rate: float,
        train_dataloader_len: int,
        num_epochs: int,
        grad_accumulation_steps: int,
        num_gpus: int,
        downsampling_k_for_audio_encoder: int = 5,
        lora_alpha: int = 128,
        lora_r: int = 64,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = ASRWithLLMModule(
            pretrained_audio_encoder_name_or_path,
            pretrained_text_decoder_name_or_path,
            downsampling_k_for_audio_encoder,
            lora_alpha=lora_alpha,
            lora_r=lora_r,
            lora_dropout=lora_dropout,
        )

    def configure_optimizers(self) -> dict:
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.get("learning_rate", 1e-4),
        )
        total_train_steps = (
            self.hparams["train_dataloader_len"]
            * self.hparams["num_epochs"]
            // (self.hparams["num_gpus"] * self.hparams["grad_accumulation_steps"])
        )
        num_warmup_steps = int(total_train_steps * 0.2)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_train_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_output_and_return_loss(self, batch: dict) -> torch.Tensor:
        output = self.model.forward(
            input_features=batch["input_features"],
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )

        return cast(torch.FloatTensor, output.loss)

    def training_step(self, batch: dict) -> torch.Tensor:
        loss = self.get_output_and_return_loss(batch)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: dict) -> None:
        loss = self.get_output_and_return_loss(batch)

        self.log(
            "valid_loss",
            loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

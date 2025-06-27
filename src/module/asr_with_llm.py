from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.whisper.modeling_whisper import WhisperModel

from src.module.projector import Projector


class ASRWithLLMModule(nn.Module):
    def __init__(
        self,
        pretrained_audio_encoder_name_or_path: str | Path,
        pretrained_text_decoder_name_or_path: str | Path,
        downsampling_k_for_audio_encoder: int,
        lora_alpha: int,
        lora_r: int,
        lora_dropout: float,
    ) -> None:
        super().__init__()

        self.audio_encoder = WhisperModel.from_pretrained(
            pretrained_audio_encoder_name_or_path, torch_dtype=torch.bfloat16
        ).encoder
        self.audio_encoder._freeze_parameters()

        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            pretrained_text_decoder_name_or_path, torch_dtype=torch.bfloat16
        )

        self.projector = Projector(
            audio_encoder_hidden_size=self.audio_encoder.config.hidden_size,
            text_decoder_hidden_size=self.text_decoder.config.hidden_size,
            downsampling_k=downsampling_k_for_audio_encoder,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.text_decoder = get_peft_model(self.text_decoder, peft_config=lora_config)
        self.text_decoder.print_trainable_parameters()

    def forward(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> CausalLMOutputWithPast:
        text_decoder = self.text_decoder.model

        embedding_audio = self.audio_encoder(
            input_features=input_features
        ).last_hidden_state
        projected_embedding_audio = self.projector(embedding_audio)

        embedding_text = text_decoder.model.embed_tokens(input_ids)

        hidden_states = text_decoder.model(
            inputs_embeds=torch.cat((projected_embedding_audio, embedding_text), dim=1)
        )[0]

        text_seq_len = input_ids.shape[1]
        logits = self.text_decoder.lm_head(hidden_states[:, -text_seq_len:, :])

        loss = text_decoder.loss_function(
            logits=logits, labels=labels, vocab_size=text_decoder.vocab_size
        )

        return CausalLMOutputWithPast(loss, logits, hidden_states=hidden_states)


if __name__ == "__main__":
    model = ASRWithLLMModule(
        "openai/whisper-small",
        "sbintuitions/sarashina2.2-1b-instruct-v0.1",
        downsampling_k_for_audio_encoder=5,
        lora_alpha=128,
        lora_r=64,
        lora_dropout=0.05,
    )

import torch
import torch.nn as nn
from einops import rearrange


class Projector(nn.Module):
    def __init__(
        self,
        audio_encoder_hidden_size: int,
        text_decoder_hidden_size: int,
        downsampling_k: int,
    ) -> None:
        super().__init__()
        self.downsampling_k = downsampling_k
        rearranged_dim = audio_encoder_hidden_size * self.downsampling_k

        self.projector = nn.Sequential(
            nn.Linear(rearranged_dim, 2048, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(2048, text_decoder_hidden_size, dtype=torch.bfloat16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        new_seq_len = (seq_len // self.downsampling_k) * self.downsampling_k

        x = x[:, :new_seq_len, :]

        # ダウンサンプリングと concat
        x = rearrange(x, "b (s k) d -> b s (k d)", k=self.downsampling_k)

        x = self.projector(x)
        return x

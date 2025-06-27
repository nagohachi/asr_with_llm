import random
import unicodedata
from pathlib import Path
from typing import Any, cast

import librosa
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchaudio.transforms import (
    FrequencyMasking,
    SpeedPerturbation,
    TimeMasking,
)
from transformers import WhisperFeatureExtractor
from transformers.tokenization_utils import PreTrainedTokenizer


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


class ASRDataset(Dataset):
    def __init__(
        self,
        data_tsv_paths: list[Path],
        tokenizer: PreTrainedTokenizer,
        feature_extractor: WhisperFeatureExtractor,
        augment_data: bool,
        augment_data_prob: float | None = None,
    ) -> None:
        super().__init__()
        self.df = pd.concat(
            [pd.read_csv(tsv_path, sep="\t") for tsv_path in data_tsv_paths]
        )
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        self.augment_data = augment_data
        self.augment_data_prob = augment_data_prob
        if self.augment_data:
            self.speedperturb = SpeedPerturbation(
                orig_freq=self.feature_extractor.sampling_rate, factors=[0.9, 1.0, 1.1]
            )
            self.specaug = nn.Sequential(
                FrequencyMasking(freq_mask_param=80), TimeMasking(time_mask_param=40)
            )
            assert self.augment_data_prob is not None

        random.seed(42)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.augment_data:
            augment = random.random() < cast(float, self.augment_data_prob)
        else:
            augment = False

        row = self.df.iloc[idx]

        waveform, sampling_rate = librosa.load(row["path"], mono=True)

        if sampling_rate != 16_000:
            waveform = librosa.resample(
                waveform, orig_sr=sampling_rate, target_sr=16_000
            )

        waveform = torch.from_numpy(waveform)

        # data augmentation
        if augment:
            waveform, _ = self.speedperturb(waveform)

        mel_feature = self.feature_extractor(
            waveform,  # type: ignore
            return_tensors="pt",
            sampling_rate=self.feature_extractor.sampling_rate,
        ).input_features[0]

        if augment:
            mel_feature = self.specaug(mel_feature)

        label = row["transcription"]
        encoded_label = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "この音声を書き起こしてください。"},
                {"role": "assistant", "content": normalize_text(label)},
            ],
            return_dict=True,
            return_tensors="pt",
        )

        return {
            "input_features": mel_feature,
            "input_ids": encoded_label["input_ids"][0],  # type: ignore
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from transformers.tokenization_utils import PreTrainedTokenizer

    from src.constants.paths import INPUT_DIR

    def get_audio_encoder_feature_extractor(
        pretrained_model_name_or_path: str | Path,
    ) -> WhisperFeatureExtractor:
        return WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)

    def get_text_decoder_tokenizer(
        pretrained_model_name_or_path: str | Path,
    ) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    csj_train_tsv_path = INPUT_DIR / "train.tsv"
    csj_eval1_tsv_path = INPUT_DIR / "eval1.tsv"
    csj_eval2_tsv_path = INPUT_DIR / "eval2.tsv"
    csj_eval3_tsv_path = INPUT_DIR / "eval3.tsv"

    audio_encoder_model_name = "openai/whisper-small"
    text_decoder_model_name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"

    audio_encoder_feature_extractor = get_audio_encoder_feature_extractor(
        audio_encoder_model_name
    )
    text_decoder_tokenizer = get_text_decoder_tokenizer(text_decoder_model_name)

    dataset = ASRDataset(
        [csj_train_tsv_path],
        text_decoder_tokenizer,
        audio_encoder_feature_extractor,
        augment_data=True,
        augment_data_prob=0.3,
    )

    for i in range(5):
        dataset[i]

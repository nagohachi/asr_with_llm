from dataclasses import dataclass

import torch
from transformers import WhisperFeatureExtractor
from trl import DataCollatorForCompletionOnlyLM


@dataclass
class ASRDataCollatorWithPadding:
    feature_extractor: WhisperFeatureExtractor
    text_data_collator: DataCollatorForCompletionOnlyLM

    def __call__(self, features: list[dict]) -> dict:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batched_input_features = self.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        batched_input_text = self.text_data_collator(input_ids, return_tensors="pt")

        return {
            "input_features": batched_input_features["input_features"].to(
                torch.bfloat16
            ),
            "input_ids": batched_input_text["input_ids"],
            "labels": batched_input_text["labels"],
        }

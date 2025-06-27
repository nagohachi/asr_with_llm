from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM

from src.data.data_collator import ASRDataCollatorWithPadding
from src.data.dataset import ASRDataset


class LitASRWithLLMDataModule(LightningDataModule):
    def __init__(
        self,
        train_tsv_paths: list[Path],
        valid_tsv_paths: list[Path],
        tokenizer: PreTrainedTokenizer,
        feature_extractor: WhisperFeatureExtractor,
        train_batch_size: int,
        valid_batch_size: int,
    ) -> None:
        super().__init__()

        self.train_dataset = ASRDataset(
            train_tsv_paths,
            tokenizer,
            feature_extractor,
            augment_data=True,
            augment_data_prob=0.3,
        )

        self.valid_dataset = ASRDataset(
            valid_tsv_paths,
            tokenizer,
            feature_extractor,
            augment_data=False,
        )

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        text_data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, response_template="<|assistant|>"
        )
        self.data_collator = ASRDataCollatorWithPadding(
            feature_extractor, text_data_collator
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=8,
            pin_memory=True,
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from transformers.tokenization_utils import PreTrainedTokenizer

    from src.constants.paths import INPUT_DIR

    csj_train_tsv_path = INPUT_DIR / "train.tsv"
    csj_eval1_tsv_path = INPUT_DIR / "eval1.tsv"
    csj_eval2_tsv_path = INPUT_DIR / "eval2.tsv"
    csj_eval3_tsv_path = INPUT_DIR / "eval3.tsv"

    audio_encoder_model_name = "openai/whisper-small"
    text_decoder_model_name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"

    def get_audio_encoder_feature_extractor(
        pretrained_model_name_or_path: str | Path,
    ) -> WhisperFeatureExtractor:
        return WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)

    def get_text_decoder_tokenizer(
        pretrained_model_name_or_path: str | Path,
    ) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    audio_encoder_feature_extractor = get_audio_encoder_feature_extractor(
        audio_encoder_model_name
    )
    text_decoder_tokenizer = get_text_decoder_tokenizer(text_decoder_model_name)

    datamodule = LitASRWithLLMDataModule(
        train_tsv_paths=[csj_train_tsv_path],
        valid_tsv_paths=[csj_eval1_tsv_path],
        tokenizer=text_decoder_tokenizer,
        feature_extractor=audio_encoder_feature_extractor,
        train_batch_size=4,
        valid_batch_size=4,
    )

    for i, data in enumerate(datamodule.train_dataloader()):
        if i >= 5:
            break

        print(data)

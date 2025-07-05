import datetime
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers.tokenization_utils import PreTrainedTokenizer

from src.constants.paths import INPUT_DIR
from src.lightning_utils.lightning_data_module import LitASRWithLLMDataModule
from src.lightning_utils.lightning_module import LitASRWithLLMModule


def get_audio_encoder_feature_extractor(
    pretrained_model_name_or_path: str | Path,
) -> WhisperFeatureExtractor:
    return WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)


def get_text_decoder_tokenizer(
    pretrained_model_name_or_path: str | Path,
) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path)


class CONFIG:
    train_batch_size = 4
    valid_batch_size = 4
    learning_rate = 1e-4
    num_epochs = 16
    grad_accumulation_steps = 2
    gpus = [0, 1, 2, 3]


def train() -> None:
    csj_train_tsv_path = INPUT_DIR / "train.tsv"
    csj_eval1_tsv_path = INPUT_DIR / "eval1.tsv"

    audio_encoder_model_name = "openai/whisper-small"
    text_decoder_model_name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"

    audio_encoder_feature_extractor = get_audio_encoder_feature_extractor(
        audio_encoder_model_name
    )
    text_decoder_tokenizer = get_text_decoder_tokenizer(text_decoder_model_name)

    datamodule = LitASRWithLLMDataModule(
        train_tsv_paths=[csj_train_tsv_path],
        valid_tsv_paths=[csj_eval1_tsv_path],
        tokenizer=text_decoder_tokenizer,
        feature_extractor=audio_encoder_feature_extractor,
        train_batch_size=CONFIG.train_batch_size,
        valid_batch_size=CONFIG.valid_batch_size,
    )

    module = LitASRWithLLMModule(
        pretrained_audio_encoder_name_or_path=audio_encoder_model_name,
        pretrained_text_decoder_name_or_path=text_decoder_model_name,
        learning_rate=CONFIG.learning_rate,
        train_dataloader_len=len(datamodule.train_dataloader()),
        num_epochs=CONFIG.num_epochs,
        grad_accumulation_steps=CONFIG.grad_accumulation_steps,
        num_gpus=len(CONFIG.gpus),
    )

    trainer = Trainer(
        devices=CONFIG.gpus,
        logger=WandbLogger(
            project="asr_with_llm",
            name=datetime.datetime.now().strftime(format="%Y%m%d_%H%M"),
        ),
        callbacks=[
            ModelCheckpoint(monitor="valid_loss", save_top_k=10, save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        max_epochs=CONFIG.num_epochs,
        num_sanity_val_steps=0,
        val_check_interval=0.5,
        log_every_n_steps=50,
        accumulate_grad_batches=CONFIG.grad_accumulation_steps,
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    train()

from argparse import ArgumentParser
from pathlib import Path

import jiwer
import torch
from tqdm.auto import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from src.constants.paths import INPUT_DIR
from src.data.dataset import ASRDataset, normalize_text
from src.lightning_utils.lightning_module import LitASRWithLLMModule
from src.module.asr_with_llm import ASRWithLLMModule
from src.scripts.train import (
    get_audio_encoder_feature_extractor,
    get_text_decoder_tokenizer,
)


def query(
    model: ASRWithLLMModule, mel_feature: torch.Tensor, tokenizer: PreTrainedTokenizer
) -> str:
    with torch.autocast(device_type="cuda"):
        embedding_audio = model.audio_encoder(
            input_features=mel_feature[None, :].to("cuda")
        ).last_hidden_state
        projected_embedding_audio = model.projector(embedding_audio)
        embedding_text = model.text_decoder.model.model.embed_tokens(
            tokenizer(
                "<|user|>音声を文字起こししてください。</s><|assistant|>",
                return_tensors="pt",
            )
            .to("cuda")
            .input_ids
        )

        all_embed = torch.cat((projected_embedding_audio, embedding_text), dim=1)
        output = model.text_decoder.generate(
            inputs_embeds=all_embed, max_new_tokens=256
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def test(model_checkpoint: Path) -> None:
    csj_eval1_tsv_path = INPUT_DIR / "eval1.tsv"
    csj_eval2_tsv_path = INPUT_DIR / "eval2.tsv"
    csj_eval3_tsv_path = INPUT_DIR / "eval3.tsv"

    audio_encoder_model_name = "openai/whisper-small"
    text_decoder_model_name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"

    module = LitASRWithLLMModule.load_from_checkpoint(model_checkpoint)
    model = module.model.to("cuda")

    feature_extractor = get_audio_encoder_feature_extractor(audio_encoder_model_name)
    tokenizer = get_text_decoder_tokenizer(text_decoder_model_name)

    model.text_decoder.generation_config.pad_token_id = tokenizer.pad_token_id

    for tsv_path in [
        csj_eval1_tsv_path,
        csj_eval2_tsv_path,
        csj_eval3_tsv_path,
    ]:
        dataset = ASRDataset(
            data_tsv_paths=[tsv_path],
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            augment_data=False,
        )

        hypotheses = []
        references = []

        for i in tqdm(range(len(dataset))):
            mel_feature = dataset[i]["input_features"]
            reference = normalize_text(dataset.df.iloc[i]["transcription"])

            hypothesis = query(
                model=model, mel_feature=mel_feature, tokenizer=tokenizer
            )

            hypotheses.append(hypothesis)
            references.append(reference)

        print(
            f"CER fpr {tsv_path.name}: {jiwer.cer(reference=references, hypothesis=hypotheses)}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt")
    args = parser.parse_args()

    test(args.ckpt)

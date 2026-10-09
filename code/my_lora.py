import itertools
import logging
from typing import Literal, Iterable, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from my_utils import Message, genTokAndMasks

T_PEFT_TYPE = Literal['lora', 'qlora4', 'qlora8']

# Hugging Face causal language models use -100 as the ignored label value.
IGNORED_LABEL = -100

logger = logging.Logger(__name__)

def loadModelForLoRA(*, model_id: str, device: torch.device, peft_type: T_PEFT_TYPE = 'lora', lora_config: LoraConfig):
    # Pre checks
    if peft_type.startswith('qlora'):
        assert device.type == 'cuda', "CUDA GPU required for quantized bitsandbytes"

    # Tokenizer load
    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #if tokenizer.pad_token_id is None:
    #    logger.warning('Tokenizer has no padding token, EOS will be used instead, could create problems with libraries (like `DataCollatorForLanguageModeling(mlm=False)`) may remove all EOS from attentions due to this')
    #    tokenizer.pad_token = tokenizer.eos_token

    # Model load
    if peft_type == 'qlora4':
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif peft_type == 'qlora8':
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None
    logger.info('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map={ '': device }, # Load directly into device avoiding too much CPU RAM usage
        dtype=torch.float16,
    )
    model.config.use_cache = False

    # Quantization aware LoRA preparations
    if peft_type.startswith('qlora'):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={ 'use_reentrant': False }) # New PyTorch support

    model = get_peft_model(model, lora_config)
    if True:
        #model.print_trainable_parameters()
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")

    return tokenizer, model

def chatAssistantOnlyMapper(tokenizer, *, device, chat_template_kwargs: dict[str, Any]|None = None):
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    def mapper(prompt: list[Message], answer: str):
        full_ids, prompt_len = genTokAndMasks(tokenizer, prompt, answer, **chat_template_kwargs)
        labels = (
            [IGNORED_LABEL] * prompt_len
            + full_ids[prompt_len:]
        )

        assert(any(x != IGNORED_LABEL for x in labels))

        return {
            'input_ids': torch.tensor(
                full_ids, dtype=torch.long, device=device,
            ),
            'attention_mask': torch.tensor(
                [1] * len(full_ids), dtype=torch.long, device=device,
            ),
            'labels': torch.tensor(
                labels, dtype=torch.long, device=device,
            ),
        }
    return mapper

def paddingDatasetBatcher(tokenizer, dataset, batch_size: int = 4, cycle: bool = False):
    pad = tokenizer.pad_token_id
    assert pad is not None, "Tokenizer has no padding token"

    if cycle:
        dataset = itertools.cycle(dataset)
    for batch in itertools.batched(dataset, batch_size):
        input_ids = pad_sequence(
            [x["input_ids"] for x in batch],
            batch_first=True,
            padding_value=pad,
        )

        attention_mask = pad_sequence(
            [x["attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0,
        )

        labels = pad_sequence(
            [x["labels"] for x in batch],
            batch_first=True,
            padding_value=-100,
        )

        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

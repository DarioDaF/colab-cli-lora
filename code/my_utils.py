from typing import TypedDict, Literal, TYPE_CHECKING

class Message(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str

#enable_thinking=False

def genTokAndMasks(tokenizer, prompt: list[Message], model_response: str, **kwargs):
    #if TYPE_CHECKING:
    #    from transformers import BatchEncoding

    # Runs the tokenization twice, but can't do differently for a random model chat template?
    # @TODO: In future if prefill is used just try to make it like real generation where the continuation is generated AFTER the model (needs to identify handoff from LLM convo correctly)

    continue_final_message = prompt[-1]['role'] == 'assistant' # Handle prefill case (this is more likely to cause a token mismatch in full pass)

    if continue_final_message:
        prompt_messages = prompt
        full_messages = prompt[:-1] + [{ 'role': 'assistant', 'content': prompt[-1]['content'] + model_response }]
    else:
        prompt_messages = prompt
        full_messages = prompt + [{ 'role': 'assistant', 'content': model_response }]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        **kwargs,
        tokenize=True,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message,
        return_dict=False,
        padding=False, # Ensure attention mask is all 1s
        truncation=False,
    )
    prompt_len = len(prompt_ids)

    full_ids = tokenizer.apply_chat_template(
        full_messages,
        **kwargs,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=False,
        return_dict=False,
        padding=False, # Ensure attention mask is all 1s
        truncation=False,
    )

    if full_ids[:prompt_len] != prompt_ids:
        raise ValueError('Full message tokenization merges prompt with answer')

    return full_ids, prompt_len

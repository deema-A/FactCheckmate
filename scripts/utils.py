# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import jsonlines
import random
import numpy as np
import transformers

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

script_dir = os.path.dirname(os.path.abspath('__file__'))

with open(os.path.join(script_dir, "config.json")) as f:
    datasets = json.load(f)["datasets"]


def load_auto_model_and_tokenizer(model_id, device="cuda"): #, torch_dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id)

    model.config._attn_implementation = "eager"
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.to(device)
    model.eval()

    torch.cuda.empty_cache()
    return model, tokenizer


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(self, sequence, tokenizer, initial_decoder_input_length: int, batch_size: int):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(tokenizer, stop_sequences, initial_decoder_input_length, batch_size):
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def prompt_model(model, tokenizer, prompt, stop, device="cuda"):
    
    # Clear the cache to free up memory and resources
    torch.cuda.empty_cache()
    
    # Prepare inputs for the model
    encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in encoding.items() if key in ["input_ids", "attention_mask"]}

    stopping_criteria = stop_sequences_criteria(tokenizer, stop, inputs['input_ids'].shape[1], inputs['input_ids'].shape[0])

    # Generate outputs using the model
    output = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        repetition_penalty=1.8,
        min_new_tokens=1,
        return_dict_in_generate=True,
        max_new_tokens=250,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id
    )

    decoded_output = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    
    # remove_whitespace
    decoded_output = decoded_output.lstrip()
    
    # Clear the cache again to ensure no leftover state
    torch.cuda.empty_cache()
    return decoded_output


def generate_hs(model, tokenizer, prompt, previous_response=None, stop=None, device="cuda", check=True, fct=False):
    torch.cuda.empty_cache()
    
    encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in encoding.items() if key in ["input_ids", "attention_mask"]}
    # inputs = {key: val for key, val in encoding.items() if key in ["input_ids", "attention_mask"]}

    stopping_criteria = stop_sequences_criteria(tokenizer, stop, inputs['input_ids'].shape[1], inputs['input_ids'].shape[0])
    max_new_tokens=1 if fct else 250
    
    # Generate outputs
    output = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        repetition_penalty=1.8,
        min_new_tokens=1,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id
    )

    if check:
        # Decode the full output to find the last complete sentence
        decoded_output = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        decoded_output = decoded_output.lstrip()
        
        # Check if the generated text matches the expected previous response
        if decoded_output.replace(" ", "").strip() != previous_response.replace(" ", "").strip():
            print("previous: ",  previous_response.strip())
            print("current: ", decoded_output.strip())
            raise ValueError("The generated output does not equal the passed answer.")

    # Extract hidden states for the length of the last complete sentence
    sequences = output.sequences[0]
    sentence_end_idx = len(sequences)
    layers = len(output.hidden_states[0])
    hs = [[] for _ in range(layers)]
    sentence_end_idx -= len(encoding["input_ids"][0])

    assert len(encoding["input_ids"][0]) == output.hidden_states[0][0].shape[1]

    if sentence_end_idx == 0:
        print("response_end_idx = 0")
        return None, None
    
    for l in range(layers):
        for t in range(sentence_end_idx):
            tokens = list(torch.unbind(output.hidden_states[t][l], dim=1))
            hs[l].extend(tokens)

    torch.cuda.empty_cache()
    return hs, len(encoding["input_ids"][0])


def load_json_file(file_path):
    """Load data from a JSON file and return it as a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

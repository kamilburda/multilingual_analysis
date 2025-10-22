import logging
import os
import random

import click
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def Prompting(model, tokenizer, prompt, candidate_premature_layers):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    hidden_states, outputs, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':1, 'candidate_premature_layers':candidate_premature_layers})
    hidden_embed = {}
    for i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
        # knowledge_neurons_word[early_exit_layer] = tokenizer.decode(knowledge_neurons[early_exit_layer][0])
        # hidden_info[early_exit_layer] = tokenizer.decode(torch.tensor(hidden_values[early_exit_layer]).to("cuda"))
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys


@click.command()
@click.option(
    "--language-corpus",
    default="english",
    help="Language for which to load a corpus.",
)
@click.option(
    "--corpus-sample-size",
    default=1000,
    help="Number of documents to sample from the corpus given by --language-corpus.",
)
@click.option(
    "--model-name",
    default="mistralai/Mistral-7B-Instruct-v0.2",
    help="Name of the model for which to detect language-specific neurons.",
)
@click.option(
    "--random-seed",
    default=112,
    help=(
        "Fixed random seed for sampling from the corpus given by --language-corpus."
        " If you do not wish to use a fixed random seed, use -1."
    ),
)
def main(
    language_corpus,
    corpus_sample_size,
    model_name,
    random_seed,
):
    logging.basicConfig(level=logging.INFO)

    if random_seed != -1:
        random.seed(random_seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    lines = []

    file_path = os.path.join("corpus_all", language_corpus + ".txt")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    lines = random.sample(lines, corpus_sample_size)

    candidate_premature_layers = []
    for i in range(32):
        candidate_premature_layers.append(i)

    activate_keys_set_fwd_up = []
    activate_keys_set_fwd_down = []
    activate_keys_set_q = []
    activate_keys_set_k = []
    activate_keys_set_v = []

    count = 0

    for prompt in tqdm(lines):
        try:
            hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, _, _ = Prompting(
                model,
                tokenizer,
                prompt,
                candidate_premature_layers,
            )
            activate_keys_set_fwd_up.append(activate_keys_fwd_up)
            activate_keys_set_fwd_down.append(activate_keys_fwd_down)
            activate_keys_set_q.append(activate_keys_q)
            activate_keys_set_k.append(activate_keys_k)
            activate_keys_set_v.append(activate_keys_v)
        except Exception as e:
            count += 1
            # Handle the OutOfMemoryError here
            print(count)
            print(e)

    # Initialize dictionary for common elements
    common_elements_dict_fwd_up = {}
    common_elements_dict_fwd_down = {}
    common_elements_dict_q = {}
    common_elements_dict_k = {}
    common_elements_dict_v = {}

    # Iterate through the keys of the first dictionary
    for key in activate_keys_set_fwd_up[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_up):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_up]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_up[key] = common_elements
    # print(common_elements_dict_fwd_up)

    for key in activate_keys_set_fwd_down[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_down):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_down]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_down[key] = common_elements
    # print(common_elements_dict_fwd_down)

    for key in activate_keys_set_q[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_q):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_q]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_q[key] = common_elements
    # print(common_elements_dict_q)

    for key in activate_keys_set_k[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_k):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_k]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_k[key] = common_elements
    # print(common_elements_dict_k)

    for key in activate_keys_set_v[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_v):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_v]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_v[key] = common_elements
    # print(common_elements_dict_v)

    file_path = os.path.join(
        "output_neurons",
        *model_name.split('/'),
        language_corpus,
        "gsm_2000_12000_" + str(corpus_sample_size - count) + ".txt",
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        file.write(str(common_elements_dict_fwd_up) + '\n')
        file.write(str(common_elements_dict_fwd_down) + '\n')
        file.write(str(common_elements_dict_q) + '\n')
        file.write(str(common_elements_dict_k) + '\n')
        file.write(str(common_elements_dict_v) + '\n')


if __name__ == "__main__":
    main()

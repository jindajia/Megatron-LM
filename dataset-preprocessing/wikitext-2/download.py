from datasets import load_dataset
from datasets import disable_caching
from datasets import load_dataset_builder
from huggingface_hub import hf_hub_download
import os
import json

save_path = "" # TODO set saving path

disable_caching()
# load dataset
train_data = load_dataset('wikitext', 'wikitext-2-v1', split='train')

# conver it to json
dataset_output_name = 'wikitext-2-v1.json'
train_data.to_json(os.path.join(save_path, dataset_output_name),  lines=True)

hf_hub_download(repo_id="gpt2", filename="merges.txt", local_dir=save_path)
hf_hub_download(repo_id="gpt2", filename="vocab.json", local_dir=save_path)
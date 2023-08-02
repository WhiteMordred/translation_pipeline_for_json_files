import os
import json
import torch
import torch.multiprocessing as mp
from transformers import MarianMTModel, MarianTokenizer
from colorama import Fore, init

def split_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(src_dir):
        if filename.endswith('.json'):
            with open(os.path.join(src_dir, filename), 'r') as f:
                data = json.load(f)
                num_files = len(data)
                num_files_per_gpu = (num_files + mp.cpu_count() - 1) // mp.cpu_count()

                for i in range(0, num_files, num_files_per_gpu):
                    split_data = data[i:i + num_files_per_gpu]
                    split_filename = os.path.join(dest_dir, f'{os.path.splitext(filename)[0]}_{i // num_files_per_gpu}.json')
                    with open(split_filename, 'w') as split_file:
                        json.dump(split_data, split_file)

def translate_text(text, model, tokenizer):
    if not text:
        return text    
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized_text.input_ids.to(model.device)
    attention_mask = tokenized_text.attention_mask.to(model.device)
    translated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
    translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return translated_text

def translate_file(i, filename, dest_dir, device, batch_size, total_files):
    print(f"{Fore.BLUE}Process {i} {Fore.YELLOW}started")
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    with open(filename, 'r') as f:
        data = json.load(f)

    for entry in data:
        for key, value in entry.items():
            if isinstance(value, str):
                value = value.strip()
            else:
                value = ""
            entry[key] = translate_text(value, model, tokenizer)

    translated_filename = os.path.join(dest_dir, os.path.basename(filename))
    with open(translated_filename, 'w') as f:
        json.dump(data, f)

    print(f"{Fore.BLUE}Process {i} {Fore.GREEN}finished")
    print(f"{Fore.MAGENTA}Progress: {((i + 1) / total_files) * 100:.2f}%")

def translate_files(src_dir, dest_dir, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{Fore.CYAN}Using device: {device}")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    files = [os.path.join(src_dir, filename) for filename in os.listdir(src_dir) if filename.endswith('.json')]
    num_processes = min(len(files), torch.cuda.device_count())

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(translate_file, [(i, filename, dest_dir, device, batch_size, len(files)) for i, filename in enumerate(files)])

def merge_files(src_dir, dest_file):
    data = []
    for filename in os.listdir(src_dir):
        if filename.endswith('.json'):
            with open(os.path.join(src_dir, filename), 'r') as f:
                data.extend(json.load(f))
    with open(dest_file, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    src_dir = './json_src'
    split_dir = './json_split'
    split_trad_dir = './json_split_trad'
    merge_file = './translated_data.json'

    print(f"{Fore.GREEN}Splitting files...")
    split_files(src_dir, split_dir)

    print(f"{Fore.GREEN}Translating files...")
    translate_files(split_dir, split_trad_dir, batch_size=4096)

    print(f"{Fore.GREEN}Merging files...")
    merge_files(split_trad_dir, merge_file)

    print(f"{Fore.GREEN}Translation completed.")

import os
import pdb
import yaml
import re
import csv
import json
import sys
import torch
import pickle
sys.path.append("/localhome/junyi/Projects/RAG-Star/mcts")


def load_prompt_template(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = yaml.safe_load(f)
        return prompt


def extract_answer(str):
    matches = re.findall(r"\\boxed{(.*?)}", str)
    if matches:
        return matches[-1]
    else:
        return "None"


def save_results(result_list, output_file):
    with open(output_file, "w") as f:
        json.dump(result_list, f, indent=4)


def save_one_result(predict_file, sample, idx, pred_answer, documents, trajectory):
    with open(predict_file, "a", encoding="utf-8") as f:
        sample["idx"] = idx
        sample["documents"] = documents
        sample["prediction"] = pred_answer
        sample["trajectory"] = trajectory
        f.write(json.dumps(sample, ensure_ascii=False, indent=4) + "\n")
        f.flush()


def read_data(file_path, start_idx, end_idx):
    with open(file_path) as f:
        data = json.load(f)
        data = data[start_idx:end_idx]
    return data


def read_wiki():
    # the wikipedia split from DPR repo: https://github.com/facebookresearch/DPR
    DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV")
    wiki_docs = []
    with open(DPR_WIKI_TSV) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            wiki_docs.append(row[2] + "\n" + row[1])

    # load GTR embedding for the Wikipedia
    GTR_EMB = os.environ.get("GTR_EMB")
    with open(GTR_EMB, "rb") as f:
        wiki_embeds = pickle.load(f)
    wiki_embeds = torch.tensor(wiki_embeds, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")

    return wiki_docs, wiki_embeds


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

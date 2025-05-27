import argparse
import collections
import json
import re
import string
import torch
import copy
import sys
import logging
import numpy as np
from nltk import sent_tokenize
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from utils.utils import normalize_answer, get_max_memory, remove_citations

QA_MODEL = "roberta-large-squad"
AUTOAIS_MODEL = "t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None


def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """
    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """

    def _rouge_calculation(hypotheses, references1, references2=[], metrics=['rougeLsum']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    logging.info("Computing ROUGE...")
    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None:  # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores['rougeLsum']


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM
    """
    print("Computing string exact match...")
    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        print("Warning: no QA pairs found in data to compute string EM")
        return 0, 0

    acc = []
    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["prediction"]))
        acc.append(np.mean(loc_acc))

    return 100 * np.mean(acc)


def compute_length(data):
    """Compute average length of predictions."""
    logging.info("Computing average length...")
    res, cntr = 0, 0
    for item in data:
        res += len(item["prediction"].split())
        cntr += 1
    return res / cntr


def compute_qa(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """
    print("Computing QA-based accuracy...")
    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        print("Warning: no QA pairs found in data to compute qa EM")
        return {'QA-EM': 0, 'QA-F1': 0, 'QA-Hit': 0, }

    # Load model
    print("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=0)

    # Get prediction
    em, f1, bins = [], [], []
    for item in tqdm(data):
        question = [qa_pair['question'] for qa_pair in item['qa_pairs']]
        context = item['prediction'] if len(item['prediction']) > 0 else " "
        results = qa_pipeline(question=question, context=context, handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0
        for idx, res in enumerate(results):
            answers = item["qa_pairs"][idx]["short_answers"]
            prediction = res["answer"]
            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            loc_counter += 1
        em.append(loc_em / loc_counter)
        f1.append(loc_f1 / loc_counter)
        bins.append(loc_em == loc_counter)

    return {
        'QA-EM': 100 * np.mean(em),
        'QA-F1': 100 * np.mean(f1),
        'QA-Hit': 100 * np.mean(bins)
    }


def compute_mauve(data):
    """Compute Mauve score."""

    logging.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(
            ' '.join((item['question'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(
            ' '.join((item['question'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))

    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large"
    )
    return out.mauve * 100


def run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_claims(data):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logging.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16,
                                                              device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, legacy=False, use_fast=False)

    logging.info("Computing claims NLI...")
    scores = []
    for item in tqdm(data):
        entail = 0
        claims = item["claims"]
        for claim in claims:
            entail += run_nli_autoais(item['output'], claim)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores)


def compute_autoais(data, qampari=False, at_most_citations=3):
    """
    Compute AutoAIS score.
    """
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16,
                                                              device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, legacy=False, use_fast=False)

    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0

    for item in tqdm(data):
        # get sentences by using NLTK
        sents = sent_tokenize(item['prediction'])

        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided

            # Find references
            ref = [r[1:].strip() for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id not in item['documents'].keys() for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([item['documents'][ref_id] for ref_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail = run_nli_autoais(joint_passage, target_sent)

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for ref_id in ref:
                    # condition A
                    passage = item['documents'][ref_id]
                    nli_result = run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(ref_id)
                        passage = '\n'.join([item['documents'][rid] for rid in subset_exclude])
                        nli_result = run_nli_autoais(passage, target_sent)
                        if nli_result:  # ref_id is not necessary
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0)  # len(sents))

    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_prec": 100 * np.mean(ais_scores_prec),
    }


def compute_qampari_f1(data):
    logging.info("Computing QAMPARI F1...")
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in data:
        o = item['output']
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0]  # delete empty answers
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]
        flat_answers = [item for sublist in answers for item in sublist]

        prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers)))
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0)
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="output directory")
    parser.add_argument("--dataset", type=str, default="asqa", help="options: asqa/qampari/eli5")
    parser.add_argument("--at_most_citations", type=int, default=3,
                        help="at most take this many documents (mostly for precision)")
    parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")
    args = parser.parse_args()

    data = []
    for output_file in os.listdir(args.result_dir):
        output_path = os.path.join(args.result_dir, output_file)
        with open(output_path, 'r') as fin:
            data.extend(json.load(fin))

    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['prediction'] = remove_citations(normalized_data[i]['prediction'])

    result = dict()
    # compute average length
    result['length'] = compute_length(normalized_data)
    # compute exact match
    result['str_em'] = compute_str_em(normalized_data)

    if args.dataset == "asqa":
        # result['rouge'] = compute_rouge(normalized_data)
        # result['mauve'] = compute_mauve(normalized_data)
        result.update(compute_qa(normalized_data))
        result.update(compute_autoais(data, qampari=False, at_most_citations=args.at_most_citations))
    elif args.dataset == "qampari":
        result.update(compute_qampari_f1(normalized_data))
        result.update(compute_autoais(data, qampari=True, at_most_citations=args.at_most_citations))
    elif args.dataset == "eli5":
        result['rouge'] = compute_rouge(normalized_data)
        result['mauve'] = compute_mauve(normalized_data)
        result['claims_nli'] = compute_claims(normalized_data)
        result.update(compute_autoais(data, qampari=False, at_most_citations=args.at_most_citations))
    else:
        raise ValueError("Fail to identify dataset name!")

    print(result)
    with open(args.result_file + ".score", "w") as fout:
        json.dump(result, fout, indent=4)


if __name__ == "__main__":
    main()

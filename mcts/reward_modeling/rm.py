import os
import re
import copy
import time
import torch
import random
import requests
import sglang as sgl
import torch.nn.functional as F
from collections import Counter, defaultdict
from models import OpenAIModel
from nltk import sent_tokenize
from vllm import LLM, SamplingParams
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
from utils.utils import load_prompt_template, remove_citations
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.nn import CrossEntropyLoss


class RM:
    def __init__(self, reference_model_path, reward_model_path, autoais_model_path, seed):
        self.reference_model_path = reference_model_path
        self.reward_model_path = reward_model_path
        self.autoais_model_path = autoais_model_path
        if "gpt" in reward_model_path or "4o" in reward_model_path:
            self.reward_model = OpenAIModel(
                reward_model_path,
                max_tokens=512,
                seed=seed
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.reward_model_path,
                                                           device_map="balanced_low_0",
                                                           trust_remote_code=True)
            self.reference_model = AutoModelForCausalLM.from_pretrained(self.reference_model_path,
                                                                        device_map="balanced_low_0",
                                                                        trust_remote_code=True)
            self.reward_model = AutoModelForCausalLM.from_pretrained(self.reward_model_path,
                                                                     device_map="balanced_low_0",
                                                                     trust_remote_code=True)
            self.autoais_tokenizer = AutoTokenizer.from_pretrained(self.autoais_model_path,
                                                                   device_map="balanced_low_0",
                                                                   trust_remote_code=True)
            self.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(self.autoais_model_path,
                                                                       device_map="balanced_low_0",
                                                                       trust_remote_code=True)

    def compute_quality_score(self, paragraph):
        with torch.no_grad():
            token_ids = self.tokenizer.encode(paragraph, add_special_tokens=True, return_tensors='pt')
            token_ids = token_ids.to(self.reference_model.device)
            reference_logits = self.reference_model(token_ids).logits
            token_ids = token_ids.to(self.reward_model.device)
            reward_logits = self.reward_model(token_ids).logits

            CEFunc = CrossEntropyLoss(reduction='none')
            labels = token_ids[..., 1:].contiguous().view(-1)

            # compute - log probs for policy model
            reward_logits = reward_logits[..., :-1, :].contiguous()
            vocab_size = reward_logits.size(-1)
            reward_logits = reward_logits.view(-1, vocab_size)
            log_policy_prob = -CEFunc(reward_logits, labels)

            # compute - log probs for reference model
            reference_logits = reference_logits[..., :-1, :].contiguous()
            reference_logits = reference_logits.view(-1, vocab_size).to(labels.device)
            log_reference_prob = -CEFunc(reference_logits, labels)

            token_length = token_ids.size(1) - 1

            diff = (log_policy_prob - log_reference_prob) / token_length
            score = torch.sum(diff, dim=-1)
        return score.item()

    def run_nli_autoais(self, passage, claim):
        """
        Run inference for assessing AIS between a premise and hypothesis.
        Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
        """
        input_text = "premise: {} hypothesis: {}".format(passage, claim)
        input_ids = self.autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(self.autoais_model.device)
        with torch.inference_mode():
            outputs = self.autoais_model.generate(input_ids, max_new_tokens=10)
        result = self.autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference = 1 if result == "1" else 0
        return inference

    def compute_citation_score(self, answer, references, at_most_citations=3):
        sent_total = 0
        sent_mcite = 0
        sent_mcite_support = 0
        sent_mcite_overcite = 0

        # get sentences by using NLTK
        sents = sent_tokenize(answer)

        if len(sents) == 0:
            return 0.0

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
            elif any([ref_id not in references.keys() for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([references[ref_id] for ref_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail = self.run_nli_autoais(joint_passage, target_sent)

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for ref_id in ref:
                    # condition A
                    passage = references[ref_id]
                    nli_result = self.run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(ref_id)
                        passage = '\n'.join([references[rid] for rid in subset_exclude])
                        nli_result = self.run_nli_autoais(passage, target_sent)
                        if nli_result:  # ref_id is not necessary
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        sent_total += len(sents)
        ais_scores = entail / len(sents)
        ais_scores_prec = entail_prec / total_citations if total_citations > 0 else 0  # len(sents))

        if ais_scores == 0 or ais_scores_prec == 0:
            return 0.0

        f1_score = (2 * ais_scores * ais_scores_prec) / (ais_scores + ais_scores_prec)
        return f1_score

    def get_rollout_reward(self, node):
        _, answer, references = node.get_trajectory()

        if answer == "":
            text_score = 0.1
            citation_score = 0.0
        else:
            normalized_answer = remove_citations(answer)
            text_score = self.compute_quality_score(normalized_answer)
            citation_score = self.compute_citation_score(answer, references)

        return text_score, citation_score

import os
import sys
import json
import time
import random
import argparse
import logging
from vllm import LLM
from tqdm import tqdm
from planning.agent import Agent
from searching.search import TreeSearch
from reward_modeling.rm import RM
from utils.utils import read_data, read_wiki, save_one_result, load_prompt_template, save_results
from transformers import AutoTokenizer
from gpueater.gpu_eater import occupy_gpus_mem
from models import PolicyModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["OPENAI_API_BASE"] = "xxx"


def main(args):
    logger.info("==== Hyper-Parameter Settings ====")
    logger.info(f"Policy Model: {args.policy_model_path}")
    logger.info(f"Reward Model: {args.reward_model_path}")
    logger.info(f"Reference Model: {args.reference_model_path}")
    logger.info(f"Autoais Model: {args.autoais_model_path}")
    logger.info(f"Retriever: {args.retriever}")
    logger.info(f"Query Sample: {args.query_sample}")
    logger.info(f"Answer Sample: {args.answer_sample}")
    logger.info(f"Retrieval Top-K: {args.retrieval_topk}")
    logger.info(f"Number of Simulations: {args.num_simulation}")
    logger.info(f"Max Number of Layers: {args.max_num_layers}")
    logger.info(f"Expand Probability: {args.expand_probability}")
    logger.info(f"Exploration Parameter: {args.c_param}")
    logger.info(f"Value Threshold: {args.value_threshold}")
    logger.info(f"Reflexion Threshold: {args.reflexion_threshold}")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Prompt Path: {args.prompt_path}")
    logger.info(f"Start Index: {args.start_idx}")
    logger.info(f"End Index: {args.end_idx}")
    logger.info(f"Log Path: {args.log_path}")
    logger.info(f"Save Path: {args.save_path}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"GPU Indices: {args.gpu_ids}")
    logger.info("============================")

    random.seed(args.seed)

    query_sampling_params = {
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 256,
        "n": args.query_sample,
        "seed": args.seed,
    }
    answer_sampling_params = {
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 256,
        "n": args.answer_sample,
        "seed": args.seed,
    }
    retrieval_params = {
        "retriever": args.retriever,
        "top_k": args.retrieval_topk,
    }

    # load data
    data = read_data(args.data_path, args.start_idx, args.end_idx)
    logger.info("Load {} samples from {}".format(len(data), args.data_path))

    # load prompt
    prompt = load_prompt_template(args.prompt_path)
    logger.info("Load prompt from {}".format(args.prompt_path))

    # load wikipedia and embeddings
    wiki_docs, wiki_embeds = read_wiki()
    logger.info("Load {} wikipedia documents and embeddings".format(len(wiki_docs)))

    # load policy model
    policy_model = PolicyModel(args.policy_model_path)
    logger.info("Load policy model from {}".format(args.policy_model_path))

    # load reward model
    reward_model = RM(args.reference_model_path, args.reward_model_path, args.autoais_model_path, args.seed)
    logger.info("Load reward models from {}".format(args.reward_model_path))

    # check completed samples
    completed_idx = []
    file_start = args.save_path.rfind("/")
    save_dir = args.save_path[:file_start]
    if os.path.exists(save_dir):
        for result_file in os.listdir(save_dir):
            with open(os.path.join(save_dir, result_file), "r") as f:
                finish_samples = json.load(f)
                for sam in finish_samples:
                    completed_idx.append(sam["idx"])
        logger.info("Load {} completed samples from {}".format(len(completed_idx), save_dir))
    else:
        os.makedirs(save_dir)

    mcts_results = []
    for sample in tqdm(data, desc="MCTS Running"):
        idx = sample["idx"]
        if idx in completed_idx:
            continue

        try:
            question = sample["question"]
            # docs = sample["docs"]
            initial_state = question + "\n"
            agent = Agent(
                policy_model,
                query_sampling_params,
                answer_sampling_params,
                reward_model,
                prompt,
                initial_state,
                retrieval_params,
            )

            # run MCTS
            mcts_search = TreeSearch(
                "MCTS",
                agent,
                initial_state,
                c_param=args.c_param,
                value_threshold=args.value_threshold,
                reflexion_threshold=args.reflexion_threshold,
            )

            logger.info("Run MCTS search for the {}-th sample".format(idx))
            start_time = time.time()
            print("Question: " + question)
            best_mcts_action, search_tree = mcts_search.run_search(
                num_simulations=args.num_simulation,
                retrieval_corpus=wiki_docs,
                doc_embeddings=wiki_embeds,
                strategy="max_terminal",
                max_num_layers=args.max_num_layers,
                expand_probability=args.expand_probability
            )
            logger.info("Finish MCRS search for the {}-th sample in {}s".format(idx, time.time() - start_time))

            logger.info("Best MCTS Action with value {}:\n{}".format(best_mcts_action.value(), best_mcts_action.state + best_mcts_action.candidate_answer().strip()))
            _, answer, documents = best_mcts_action.get_trajectory()
            sample["value"] = best_mcts_action.value()
            sample["documents"] = documents
            sample["prediction"] = answer
            sample["trajectory"] = best_mcts_action.state + best_mcts_action.candidate_answer()
            mcts_results.append(sample)
            logger.info(f"Finish the {idx}-th sample successfully")
        except:
            logger.exception(sys.exc_info())
            logger.info(f"Fail to complete the {idx}-th sample")

    save_results(mcts_results, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # mcts setting
    parser.add_argument("--policy_model_path", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--reference_model_path", type=str, default="Llama-3-8B-Instruct")
    parser.add_argument("--reward_model_path", type=str, default="Llama-3-8B-SFR-Iterative-DPO-R")
    parser.add_argument("--autoais_model_path", type=str, default="t5_xxl_true_nli_mixture")
    parser.add_argument("--retriever", type=str, default="gtr", help="options: bm25/gtr")
    parser.add_argument("--retrieval_topk", type=int, default=3, help="top-k documents for answering")
    parser.add_argument("--query_sample", type=int, default=5, help="number of sampling search queries")
    parser.add_argument("--answer_sample", type=int, default=1, help="number of sampling answers for each query")
    parser.add_argument("--num_simulation", type=int, default=30)

    # search setting
    parser.add_argument("--max_num_layers", type=int, default=4)
    parser.add_argument("--expand_probability", type=float, default=0.2)
    parser.add_argument("--c_param", type=float, default=0, help="co-efficient to control exploration")
    parser.add_argument("--value_threshold", type=float, default=0)
    parser.add_argument("--reflexion_threshold", type=int, default=10)

    # dataset setting
    parser.add_argument("--data_path", type=str, default="data/asqa.json")
    parser.add_argument("--prompt_path", type=str, default="prompts/asqa.yaml")
    parser.add_argument("--start_idx", type=int, default=0, help="start index of samples")
    parser.add_argument("--end_idx", type=int, default=None, help="end index of samples")

    # general setting
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ids", type=str, help="gpu indices")

    args = parser.parse_args()

    # set gpus
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        # occury gpus mem
        occupy_gpus_mem()

    # create log directories if they don't exist
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    file_handler = logging.FileHandler(args.log_path, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    main(args)

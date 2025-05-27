import os
import pdb
import time
import torch
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from searching.tree import Node
from sentence_transformers import SentenceTransformer


class Agent:
    def __init__(
            self,
            policy_model,
            query_sampling_params,
            answer_sampling_params,
            reward_model,
            prompt,
            initial_state,
            retrieval_params,
            discount_factor=0.2,
    ):
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.prompt = prompt["query-answer"]

        self.message = [
            {"role": "system", "content": self.prompt["system"]},
            {"role": "user", "content": self.prompt["user"].format(question=initial_state.strip())},
        ]

        if "gpt" in self.policy_model.get_model_name():
            self.query_sampling_params = {
                **query_sampling_params,
                "stop": ["\n"],
            }
            self.answer_sampling_params = {
                **answer_sampling_params,
                "stop": ["\n"],
            }
        else:
            self.template = self.policy_model.apply_chat_template(self.message)
            self.query_sampling_params = SamplingParams(
                temperature=query_sampling_params["temperature"],
                top_p=query_sampling_params["top_p"],
                n=query_sampling_params["n"],
                max_tokens=query_sampling_params["max_tokens"],
                stop=[
                    "\n",
                    "Document",
                    "Output:",
                    "Reflexion:",
                ],
                seed=query_sampling_params["seed"],
            )
            self.answer_sampling_params = SamplingParams(
                temperature=answer_sampling_params["temperature"],
                top_p=answer_sampling_params["top_p"],
                n=answer_sampling_params["n"],
                max_tokens=answer_sampling_params["max_tokens"],
                stop=[
                    "\n",
                    "Search:",
                    "End",
                ],
                seed=answer_sampling_params["seed"],
            )

        self.initial_state = initial_state
        self.retrieval_params = retrieval_params
        self.discount_factor = discount_factor

    def perform_querying(self, simu_iter, leaf_node, leaf_node_layer, corpus, embeds):
        if leaf_node.is_terminated():
            return []

        leaf_node_state = leaf_node.state.replace(self.initial_state, "")

        if "gpt" in self.policy_model.get_model_name():
            if leaf_node_state:
                prompt = self.message + [
                    {
                        "role": "assistant",
                        "content": (leaf_node_state + leaf_node.candidate_answer()).strip(),
                    },
                    {
                        "role": "user",
                        "content": self.prompt["querying"],
                    },
                ]
            else:
                prompt = self.message
        else:
            prompt = [
                self.template
                + leaf_node_state
                + leaf_node.candidate_answer()
            ]
        responses = self.policy_model.generate("querying", prompt, self.query_sampling_params)

        print("-" * 10 + "Querying" + "-" * 10)
        for i, res in enumerate(responses):
            print(f"Search query {i + 1}: {res.strip()}")

        querying_results = []
        print("-" * 10 + "Answering" + "-" * 10)
        for i, response in enumerate(responses):
            print(f"Node id: {simu_iter}_{i + 1}")
            print(f"Search query {i + 1}: {response.strip()}")

            if response == "":
                continue

            if response.startswith("End"):
                # search terminate
                new_state = (
                        leaf_node.state
                        + leaf_node.candidate_answer()
                        + "End"
                        + "\n"
                )
                new_node = Node(id=f"{simu_iter}_{i + 1}", parent=leaf_node, state=new_state)
                reward = leaf_node.value()
                new_node._initial_value = reward
                new_node.set_terminated()
                querying_results.append((new_node, reward))
                print("Arrive the end state!")
                continue

            # update state
            new_state = (
                    leaf_node.state
                    + leaf_node.candidate_answer()
                    + "Search: "
                    + response
            )
            # perform retrieval
            retrieved_documents = self.perform_retrieval(response.strip(), corpus, embeds,
                                                         self.retrieval_params.get("top_k"))
            for doc_id in range(len(retrieved_documents)):  # based on layer
                new_state += f"Document [{leaf_node_layer * 3 + doc_id + 1}] (Title: {retrieved_documents[doc_id]["title"]}) {retrieved_documents[doc_id]["text"]}\n"

            # perform reasoning rollouts
            answer_list = self.perform_answering(new_state)  # cannot add the prefix """Output:"""

            new_node = Node(id=f"{simu_iter}_{i + 1}", parent=leaf_node, state=new_state)
            new_node.extend_answer(answer_list)
            new_node.set_candidate_answer()

            print(f"Output sentences/thoughts: {answer_list}")
            text_score, citation_score = self.reward_model.get_rollout_reward(new_node)
            reward = text_score + citation_score
            new_node._initial_value = reward
            print(f"Text Score: {text_score}, Citation Score: {citation_score}, Overall Reward: {reward}")

            querying_results.append((new_node, reward))

        return querying_results

    def perform_answering(self, state):
        state = state.replace(self.initial_state, "")

        if "gpt" in self.policy_model.get_model_name():
            prompt = self.message + [
                {
                    "role": "assistant",
                    "content": state.strip(),
                },
                {
                    "role": "user",
                    "content": self.prompt["answering"],
                },
            ]
        else:
            prompt = [self.template + state]
        responses = self.policy_model.generate("answering", prompt, self.answer_sampling_params)
        return responses

    def perform_retrieval(self, query, docs, embeds, topk):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        RETRIEVER_PATH = os.environ.get("RETRIEVER")
        encoder = SentenceTransformer(RETRIEVER_PATH, device=device)

        with torch.inference_mode():
            query_emb = encoder.encode(query, show_progress_bar=False, normalize_embeddings=True)
            query_emb = torch.tensor(query_emb, dtype=torch.float16, device="cpu")

        encoder = encoder.to("cpu")
        del (encoder)  # save gpu mem

        # embedding of top-k docs
        # docs_text = [d["title"] + "\n" + d["text"] for d in docs]
        # docs_id = [int(d["id"]) for d in docs]
        # docs_emb = embeds[docs_id]

        query_emb = query_emb.to(device)
        # scores = torch.matmul(docs_emb, query_emb)
        scores = torch.matmul(embeds, query_emb)
        score, idx = torch.topk(scores, topk)
        query_emb = query_emb.to("cpu")
        score = score.to("cpu")
        idx = idx.to("cpu")

        ret_docs = []
        for i in range(idx.size(0)):
            id = idx[i].item() + 1
            title, text = docs[idx[i].item()].split("\n")
            # id = docs_id[idx[i].item()]
            # title, text = docs_text[idx[i].item()].split("\n")
            ret_docs.append({"id": str(id), "title": title, "text": text, "score": score[i].item()})

        # del (docs_emb)
        del (query_emb)
        del (scores)
        torch.cuda.empty_cache()

        return ret_docs

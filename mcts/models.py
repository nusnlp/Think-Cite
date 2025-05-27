import os
import pdb
from transformers import AutoTokenizer
from vllm import LLM
from openai import OpenAI
from utils.utils import extract_answer
from datetime import datetime


class PolicyModel:
    def __init__(self, model_path):
        if "gpt" in model_path:
            self.model_path = model_path
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_API_BASE"),
            )
        else:
            self.model_path = model_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = LLM(
                model_path,
                gpu_memory_utilization=0.82,
                tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")),
                max_model_len=40000,
            )

    def get_model_name(self):
        return self.model_path

    def apply_chat_template(self, message, tokenize=False, add_generation_prompt=True):
        if "gpt" in self.model_path:
            pass
        else:
            template = self.tokenizer.apply_chat_template(
                message, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
            return template

    def generate(self, stage, prompt, sampling_params):
        if "gpt" in self.model_path:
            outputs = self.client.chat.completions.create(
                model=self.model_path,
                messages=prompt,
                **sampling_params
            )
            if stage == "querying":  # querying
                responses = []
                for choice in outputs.choices:
                    content = choice.message.content
                    if content.startswith("Search:"):
                        responses.append(content.split("Search:")[-1].replace("\n", " ").strip() + "\n")
                    elif content.startswith("End"):
                        responses.append("End" + "\n")
                    else:
                        responses.append(content.replace("\n", " ").strip() + "\n")
            else:  # answering
                responses = []
                for choice in outputs.choices:
                    content = choice.message.content
                    if content.startswith("Output:"):
                        action = "Output"
                        answer_start = content.find("Output:") + len("Output:")
                    elif content.startswith("Reflexion:"):
                        action = "Reflexion"
                        answer_start = content.find("Reflexion") + len("Reflexion:")
                    else:
                        action = "Output"
                        answer_start = 0

                    answer_end = min(content.find("Search:", answer_start), content.find("End", answer_start))
                    if answer_end == -1:
                        answer_end = len(content)

                    content = (
                        content[answer_start:answer_end]
                        .replace("\n", " ")
                        .strip()
                        + "\n"
                    )
                    responses.append((action, content))
        else:
            if stage == "querying":
                outputs = self.llm.generate(prompt, sampling_params)
                responses = []
                for choice in outputs[0].outputs:
                    content = choice.text
                    if content.startswith("Search:"):
                        responses.append(content.split("Search:")[-1].replace("\n", " ").strip() + "\n")
                    elif content.startswith("End"):
                        responses.append("End" + "\n")
                    else:
                        responses.append(content.replace("\n", " ").strip() + "\n")
            else:
                outputs = self.llm.generate(prompt, sampling_params)
                responses = []
                for choice in outputs[0].outputs:
                    content = choice.text.replace("\n", " ").strip()
                    if content.startswith("Output:"):
                        action = "Output"
                        answer_start = content.find("Output:") + len("Output:")
                    elif content.startswith("Reflexion:"):
                        action = "Reflexion"
                        answer_start = content.find("Reflexion") + len("Reflexion:")
                    else:
                        action = "Output"
                        answer_start = 0

                    answer_end = min(content.find("Search:", answer_start), content.find("End", answer_start))
                    if answer_end == -1:
                        answer_end = len(content)

                    content = (
                        content[answer_start:answer_end]
                        .replace("\n", " ")
                        .strip()
                        + "\n"
                    )
                    responses.append((action, content))
        return responses
    
    def generate_batch(self, stage, batch_prompt, sampling_params):
        batch_response = []
        if "gpt" in self.model_path:
            for prompt in batch_prompt:
                outputs = self.client.chat.completions.create(
                    model=self.model_path, messages=prompt, **sampling_params
                )
                if stage == "querying":  # querying
                    responses = []
                    for choice in outputs.choices:
                        content = choice.message.content
                        if content.startswith("Search:"):
                            responses.append(content.split("Search:")[-1].replace("\n", " ").strip() + "\n")
                        elif content.startswith("End"):
                            responses.append("End" + "\n")
                        else:
                            responses.append(content.replace("\n", " ").strip() + "\n")
                else:  # answering
                    responses = []
                    for choice in outputs.choices:
                        content = choice.message.content
                        if content.startswith("Output:"):
                            action = "Output"
                            answer_start = content.find("Output:") + len("Output:")
                        elif content.startswith("Reflexion:"):
                            action = "Reflexion"
                            answer_start = content.find("Reflexion") + len("Reflexion:")
                        else:
                            action = "Output"
                            answer_start = 0

                        answer_end = min(content.find("Search:", answer_start), content.find("End", answer_start))
                        if answer_end == -1:
                            answer_end = len(content)

                        content = (
                                content[answer_start:answer_end]
                                .replace("\n", " ")
                                .strip()
                                + "\n"
                        )
                        responses.append((action, content))
                batch_response.append(responses)
        else:
            if stage == "querying":
                outputs = self.llm.generate(batch_prompt, sampling_params)
                for output in outputs:
                    responses = []
                    for choice in output.outputs:
                        content = choice.text
                        if content.startswith("Search:"):
                            responses.append(content.split("Search:")[-1].replace("\n", " ").strip() + "\n")
                        elif content.startswith("End"):
                            responses.append("End" + "\n")
                        else:
                            responses.append(content.replace("\n", " ").strip() + "\n")
                    batch_response.append(responses)
            else:
                outputs = self.llm.generate(batch_prompt, sampling_params)
                for output in outputs:
                    responses = []
                    for choice in output.outputs:
                        content = choice.text.replace("\n", " ").strip()
                        if content.startswith("Output:"):
                            action = "Output"
                            answer_start = content.find("Output:") + len("Output:")
                        elif content.startswith("Reflexion:"):
                            action = "Reflexion"
                            answer_start = content.find("Reflexion") + len("Reflexion:")
                        else:
                            action = "Output"
                            answer_start = 0

                        answer_end = min(content.find("Search:", answer_start), content.find("End", answer_start))
                        if answer_end == -1:
                            answer_end = len(content)

                        content = (
                            content[answer_start:answer_end]
                            .replace("\n", " ")
                            .strip()
                            + "\n"
                        )
                        responses.append((action, content))
                    batch_response.append(responses)
        return batch_response


class OpenAIModel:
    def __init__(
        self,
        model_name="o1-mini",
        temperature=0,
        max_tokens=4096,
        top_p=0.95,
        presence_penalty=0,
        frequency_penalty=0,
        n_reward=1,
        n_complete=1,
        max_retry=10,
        seed=42,
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE"),
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.n_reward = n_reward
        self.n_complete = n_complete
        self.max_retry = max_retry
        self.seed = seed

    def generate(self, messages):
        response = self.client.chat.completions.create(
            **{
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                # "top_p": self.top_p,
                # "presence_penalty": self.presence_penalty,
                # "frequency_penalty": self.frequency_penalty,
                # "n": self.n_reward
            }
        )
        response = response.choices[0].message.content
        return response

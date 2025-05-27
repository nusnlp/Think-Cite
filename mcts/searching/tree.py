import pdb
import re
import random


class Node(object):

    def __init__(
        self, id="root", parent: "Node" = None, state=None, initial_value: float = 0.0
    ):
        self._parent = parent
        self._children = []
        self._answer_list = []
        self._answer_reward_list = []
        self._candidate_answer = ""
        self._visit_count = 0
        self._value_sum = 0
        self._terminated = False
        self._initial_value = initial_value
        self.state = state
        # self._end = False
        self._id = id
        self._success = False

    def set_success(self):
        self._success = True

    def get_node_id(self):
        return self._id

    def value(self):
        if self._visit_count == 0:
            return self._initial_value
        return self._value_sum / self._visit_count

    def parent(self):
        return self._parent

    def children(self):
        return self._children

    def answers(self):
        return self._answer_list

    def answer_rewards(self):
        return self._answer_reward_list

    def candidate_answer(self):
        return self._candidate_answer

    def visit_count(self):
        return self._visit_count

    def is_terminated(self):
        return self._terminated

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return self._children == []

    def update(self, value: float):
        self._visit_count += 1
        self._value_sum += value

    def add_child(self, node):
        self._children.append(node)

    def extend_answer(self, answers):
        self._answer_list.extend(answers)

    def set_candidate_answer(self):
        if len(self._answer_list) == 1:
            action, content = self._answer_list[0][0], self._answer_list[0][1].strip()
            self._candidate_answer = action + ": " + content + '\n'
            # if "End" in self._candidate_answer:
            #     print("Obtain the boxed answer!")
            #     self.set_terminated()
            #     self._end = True
        else:
            print(f"set_candidate_answer_error, the len of answer list is {len(self._answer_list)}")

    def get_trajectory(self):
        trajectory = self.state + self.candidate_answer()
        queries, sentences = [], []
        documents = dict()
        for line in trajectory.split("\n"):
            if line.startswith("Search:"):
                queries.append(line[len("Search:"):].strip())
            elif line.startswith("Output:"):
                sentences.append(line[len("Output:"):].strip())
            elif line.startswith("Document"):
                doc_id = re.findall(r"\[\d+", line)[0][1:].strip()
                text = line[len("Document [" + doc_id + "]"):].strip()
                documents[doc_id] = text
            else:
                continue
        sentences = " ".join(sentences)

        return queries, sentences, documents

    def set_terminated(self):
        self._terminated = True

    def get_info(self):
        return {
            "state": self.state,
            "visit_count": self._visit_count,
            "value": self.value(),
            "terminated": self.is_terminated(),
            "answers": self.answers(),
            "answers_reward_list": self.answer_rewards(),
            "idx": self._id,
            "parent": None if self._parent is None else self._parent._id,
            "success": self._success,
        }

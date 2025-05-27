import math
import pdb
import random
import time
from searching.tree import Node


class TreeSearch:
    def __init__(self, search_method, agent, initial_state, **kwargs):
        # Initialize the search method
        self.agent = agent
        self.search_method = search_method
        self.search_algorithm = None

        # Select the search algorithm based on the search_method
        if search_method == "MCTS":
            self.search_algorithm = MCTS(self.agent, initial_state, **kwargs)
        else:
            raise ValueError(f"Unknown search method: {search_method}")

    def run_search(self, **kwargs):
        if self.search_method == "MCTS":
            return self.search_algorithm.run_search(**kwargs)


class MCTS:
    def __init__(self, agent, initial_state, c_param=0, value_threshold=0, reflexion_threshold=10):
        # Initialize with exploration parameter (c_param)
        self.agent = agent
        self.c_param = c_param  # c_param 1
        self.value_threshold = value_threshold  # TODO jjh-兜底策略选出top value的值
        self.root = Node(state=initial_state)  # Use initial_state for root node
        self.reflexion_count = 0
        self.reflexion_threshold = reflexion_threshold

    def selection(self, expand_probability=0.0):
        expand_probability = expand_probability
        decay_rate = 0.8
        # Start from the root and select the best leaf node to expand
        current_node = self.root
        selected_node = [current_node.get_node_id()]
        while not current_node.is_terminated():
            if current_node.is_leaf() or random.uniform(0, 1) < expand_probability:
                print(f"Selected node list: {str(selected_node)}")
                return current_node
            print("Node before selection:", current_node.get_node_id())
            current_node = self.select_next_action(current_node)
            selected_node.append(current_node.get_node_id())
            print("Node after selection:", current_node.get_node_id())
            expand_probability *= decay_rate
        print(f"The selected node is terminated: {current_node.get_node_id()}")
        print(f"Selected node list: {str(selected_node)}")
        return current_node

    def expansion(self, simulation_iter, leaf_node, leaf_node_layer, corpus, embeds):
        # Use the agent to perform wikipedia querying on the current leaf node's state
        # Assume perform_querying returns a list of tuples (new_node, reward)
        querying_results = self.agent.perform_querying(simulation_iter, leaf_node, leaf_node_layer, corpus, embeds)
        for new_node, value in querying_results:
            leaf_node.add_child(new_node)
            self.backpropagation(new_node, value)
            if new_node.candidate_answer().startswith("Reflexion:"):
                self.reflexion_count += 1

    def backpropagation(self, node, reward):
        # Update the value and visit count of the nodes along the path to the root
        current_node = node
        while current_node is not None:
            # Update the visit count and value with the reward
            current_node.update(reward)
            # Move up to the parent node
            current_node = current_node.parent()

    def select_next_action(self, node):
        # Select the next action based on UCB (Upper Confidence Bound)
        best_value = -float("inf")
        best_nodes = []
        for child in node.children():
            ucb_value = child.value() + self.c_param * math.sqrt(
                math.log(node.visit_count()) / (child.visit_count() + 1)
            )
            if ucb_value == best_value:
                best_nodes.append(child)
            if ucb_value > best_value:
                best_value = ucb_value
                best_nodes = [child]

        return random.choice(best_nodes)

    def ucb_value(self, node):
        """
        Calculate the UCB1 value for a node.
        """
        parent_visit_count = node.parent().visit_count() if node.parent() else 1
        return node.value() + self.c_param * math.sqrt(
            math.log(parent_visit_count) / (node.visit_count() + 1)
        )

    def best_path(
        self, node_list, strategy="most_visited", defalut_strategy="highest_value"
    ):
        """
        Select the best path from root to a leaf based on the given strategy.
        :param strategy: The strategy to use for selecting the best path.
                         Options are 'most_visited', 'highest_value', or 'ucb'.
        :return: A list of nodes representing the path from root to the best leaf.
        """
        current_node = self.root
        path = [current_node]

        if strategy == "max_terminal":
            # TODO:
            all_terminal_nodes = []
            all_nodes = [self.root]
            while all_nodes:
                node = all_nodes.pop()
                if node.is_terminated():
                    all_terminal_nodes.append(node)
                for cn in node.children():
                    all_nodes.append(cn)
            print(f"Obtain {len(all_terminal_nodes)} candidates.")
            # cjy-如果simulation结束都没有终止结点，则使用默认策略（其实这里返回的，就是找不到答案的）
            if len(all_terminal_nodes) != 0:
                answer_node = max(all_terminal_nodes, key=lambda node: node.value())
                answer_node.set_success()
                reverse_path = []
                while not answer_node.is_root():
                    reverse_path.append(answer_node)
                    answer_node = answer_node.parent()
                reverse_path.append(answer_node)
                path = list(reversed(reverse_path))
            else:
                strategy = defalut_strategy
                while not current_node.is_leaf():
                    if strategy == "most_visited":
                        # Select the child with the highest visit count
                        current_node = max(
                            current_node.children(), key=lambda node: node.visit_count()
                        )
                    elif strategy == "highest_value":
                        # Select the child with the highest average value
                        current_node = max(
                            current_node.children(), key=lambda node: node.value()
                        )
                    elif strategy == "ucb":
                        # Select the child with the highest UCB value
                        current_node = max(
                            current_node.children(),
                            key=lambda node: self.ucb_value(node),
                        )
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                    path.append(current_node)
        else:
            while not current_node.is_leaf():
                if strategy == "most_visited":
                    # Select the child with the highest visit count
                    current_node = max(
                        current_node.children(), key=lambda node: node.visit_count()
                    )
                elif strategy == "highest_value":
                    # Select the child with the highest average value
                    current_node = max(
                        current_node.children(), key=lambda node: node.value()
                    )
                elif strategy == "ucb":
                    # Select the child with the highest UCB value
                    current_node = max(
                        current_node.children(), key=lambda node: self.ucb_value(node)
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                path.append(current_node)
        self.get_tree(self.root, node_list)
        return path[-1], node_list

    def run_search(
        self,
        num_simulations,
        retrieval_corpus,
        doc_embeddings,
        strategy="highest_value",
        max_num_layers=4,
        expand_probability=0.2,
    ):
        # Run the MCTS search for a specified number of simulations
        explore_flag = False
        node_list = []
        for num in range(1, num_simulations+1):
            print(f"========Simulation {num}===========")
            if explore_flag:
                leaf_node = self.selection(expand_probability)
                explore_flag = False
            else:
                leaf_node = self.selection()

            if leaf_node.is_terminated():
                continue

            leaf_node_layer = self.get_layer_num(leaf_node)
            # TODO: 增加横向探索
            if leaf_node_layer >= max_num_layers:
                explore_flag = True
                print(f"Arrive the maximum {max_num_layers}-th layer. Skip to explore other nodes.")
                continue

            s = time.time()
            self.expansion(num, leaf_node, leaf_node_layer, retrieval_corpus, doc_embeddings)
            print(f"[Time]: expanding the node {leaf_node.get_node_id()} consumes {time.time() - s}s.")

            if self.reflexion_count > self.reflexion_threshold:
                raise Exception("Reflexion threshold exceeded")

            node_list.clear()
            best_node = None
            max_value = float("-inf")
            nodes_to_visit = [self.root]
            while nodes_to_visit:
                current_node = nodes_to_visit.pop()
                if current_node.is_terminated():
                    if current_node.value() > max_value:
                        best_node = current_node
                        max_value = current_node.value()
                nodes_to_visit.extend(current_node.children())
            if best_node is not None and best_node.value() > self.value_threshold:
                best_node.set_success()
                self.get_tree(self.root, node_list)
                return best_node, node_list

        # return self.best_path(strategy=strategy)[-1], node_list
        return self.best_path(node_list, strategy=strategy)

    def get_layer_num(self, node):
        count = 0
        while not node.is_root():
            node = node.parent()
            count += 1
        return count

    def get_tree(self, node, node_list):
        info = node.get_info()
        node_list.append(info)
        if not node.is_terminated():
            for c_child in node.children():
                self.get_tree(c_child, node_list)

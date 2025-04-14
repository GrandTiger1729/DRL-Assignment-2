from typing import Dict, List

import copy
import math
import numpy as np
from game2048env import Game2048Env
from ntuple_approximator import NTupleApproximator


class TD_MCTS_Node:
    def __init__(self, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.parent = parent
        self.action = action
        self.children: Dict[int, TD_MCTS_Node] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4)]

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10):
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth
        self.approximator: NTupleApproximator = approximator

    def select_child(self, node: TD_MCTS_Node, legal_actions: List[int]):
        def value(child: TD_MCTS_Node):
            if child.visits == 0:
                return float('inf')
            return child.total_reward / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits)
        return max([node.children[a] for a in legal_actions], key=value)
    
    def get_best_action(self, sim_env: Game2048Env, legal_moves=None):
        if not legal_moves:
            legal_moves = sim_env.get_legal_moves()
        def value(action):
            temp_env = copy.deepcopy(sim_env)
            next_state, reward, _ = temp_env.step(action)
            return reward + self.approximator.get_value(next_state)
        return max(legal_moves, key=lambda a: sum(value(a) for _ in range(3)) / 3)
    
    def evaluate(self, sim_env: Game2048Env, action, iteration=3):
        def value(action):
            temp_env = copy.deepcopy(sim_env)
            new_board, reward, _done, = temp_env.step(action)
            return reward + self.approximator.get_value(new_board)
        
        total = sum(value(action) for _ in range(iteration))
        # return sim_env.evaluate() + total / iteration
        return sim_env.score + total / iteration
    
    def rollout(self, sim_env: Game2048Env, action):
        return self.evaluate(sim_env, action)

    def backpropagate(self, node: TD_MCTS_Node, total_reward):
        while node is not None:
            node.visits += 1
            node.total_reward += total_reward
            node = node.parent

    def run_simulation(self, root: TD_MCTS_Node, env: Game2048Env):
        node = root
        sim_env = copy.deepcopy(env)
        
        # Selection
        done = sim_env.is_game_over()
        untried_actions = [i for i in node.untried_actions if sim_env.is_move_legal(i)]
        while not done and not untried_actions:
            legal_actions = sim_env.get_legal_moves()
            node = self.select_child(node, legal_actions)
            _, _, done = sim_env.step(node.action)
            untried_actions = [i for i in node.untried_actions if sim_env.is_move_legal(i)]

        # Expansion and Rollout
        if untried_actions:
            action = self.get_best_action(sim_env, untried_actions)
            node.untried_actions.remove(action)
            node.children[action] = TD_MCTS_Node(node, action)
            node = node.children[action]
            rollout_reward = self.rollout(sim_env, action)
        else:
            assert(sim_env.is_game_over())
            rollout_reward = sim_env.score # + sim_env.evaluate()
        
        # Backpropagation
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


def run_experiment(approximator: NTupleApproximator, iterations=50, exploration_constant=10, rollout_depth=0):

    env = Game2048Env()
    td_mcts = TD_MCTS(approximator, iterations=iterations, exploration_constant=exploration_constant, rollout_depth=rollout_depth)

    state = env.reset()
    done = False
    step = 0
    while not done:
        root = TD_MCTS_Node()
        for _ in range(td_mcts.iterations):
            td_mcts.run_simulation(root, env)

        best_action, distribution = td_mcts.best_action_distribution(root)
        state, reward, done = env.step(best_action)
        step += 1
        # print("Step: {}, Score: {}, Action: {}, Distribution: {}".format(step, env.score, best_action, distribution))
        if step % 100 == 0:
            print("Step: {}, Score: {}, Action: {}, Distribution: {}".format(step, env.score, best_action, distribution))

    print("Final score: {}".format(env.score))

    return env.score

if __name__ == "__main__":
    import pickle

    # with open("ntuple_approximator.pkl", 'rb') as f:
    #     approximator = pickle.load(f)
    approximator = NTupleApproximator()
    approximator.load()

    GAMES = 1
    scores = []
    for i in range(GAMES):
        score = run_experiment(approximator, iterations=50, exploration_constant=10, rollout_depth=0)
        scores.append(score)
        print("Game {}: Score {}".format(i + 1, score))

    total = sum(scores)
    print("Average score over {} games: {:.2f}".format(GAMES, total / GAMES))
    print(scores)
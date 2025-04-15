import numpy as np

from game2048env import Game2048Env
from ntuple_approximator import NTupleApproximator
from td_mcts_2048 import TD_MCTS, TD_MCTS_Node


approximator = NTupleApproximator()
approximator.load()
step_count = 0

def get_action(state, score):
    global approximator, step_count

    env = Game2048Env()
    env.board = state.copy()
    env.score = score

    # legal_moves = env.get_legal_moves()
    # assert legal_moves, "No legal moves available."

    # # Choose the best action using TD(0) greedy
    # values = []
    # for action in legal_moves:
    #     temp_env = Game2048Env()
    #     temp_env.board = env.board.copy()
    #     temp_env.score = env.score
    #     next_state, reward, _done = temp_env.step(action)
    #     values.append(reward + approximator.get_value(next_state))

    # best_action = legal_moves[np.argmax(values)]

    # return best_action

    td_mcts = TD_MCTS(approximator, iterations=10, exploration_constant=0.1, rollout_depth=0)

    root = TD_MCTS_Node()
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root, env)

    best_action, distribution = td_mcts.best_action_distribution(root)

    step_count += 1
    if step_count % 50 == 0:
        print(f"Step: {step_count}, Score: {score}, Distribution: {distribution}")
    
    return best_action
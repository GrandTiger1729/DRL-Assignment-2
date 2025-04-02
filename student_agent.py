import pickle

from game import Game2048Env
from ntuple_approximator import NTupleApproximator
from ntuple_action import get_best_action

env = Game2048Env()  # Initialize the game environment
with open('ntuple_approximator.pkl', 'rb') as f:
    approximator = pickle.load(f)

def get_action(state, score):
    global env, approximator

    env.board = state.copy()
    env.score = score
    env.cache_after_states = [None] * 4  # Reset cache for the current state

    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return None  # No legal moves available
    best_action = get_best_action(env, approximator, legal_moves)

    return best_action  # Choose the best action based on evaluation
import pickle

from game2048env import Game2048Env
from approximator import NTupleApproximator

def get_best_action(env, approximator, legal_moves, gamma=0.99):
    score = env.score
    GameEnv = type(env)
    temp_env = GameEnv()
    def value(action):
        temp_env.board = env.cache_after_states[action][0].copy()
        temp_env.add_random_tile()
        next_state = temp_env.board
        new_score = env.cache_after_states[action][1]
        if temp_env.is_game_over():
            new_score = 0
        return new_score - score + gamma * approximator.value(next_state)
    return max(legal_moves, key=value)

env = Game2048Env()  # Initialize the game environment
with open('n_tuple_approximator.pkl', 'rb') as f:
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
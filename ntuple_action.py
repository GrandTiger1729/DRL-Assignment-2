from game import Game2048Env
from ntuple_approximator import NTupleApproximator

def get_best_action(env: Game2048Env, approximator: NTupleApproximator, legal_moves, gamma=0.99):
    score = env.score
    GameEnv = type(env)
    temp_env = GameEnv()
    def value(action):
        temp_env.board = env.cache_after_states[action][0].copy()
        temp_env.add_random_tile()
        next_state = temp_env.board
        new_score = env.cache_after_states[action][1]
        if temp_env.is_game_over():
            new_score -= 10000
        return new_score - score + gamma * approximator.value(next_state)
    return max(legal_moves, key=value)
import copy

from game2048env import Game2048Env
from ntuple_approximator import NTupleApproximator

def get_td_best_action(env: Game2048Env, approximator: NTupleApproximator, legal_moves, simulation=1):
    def value(action):
        temp_env = copy.deepcopy(env)
        next_state, reward, _ = temp_env.step(action)
        return reward + approximator.get_value(next_state)
    return max(legal_moves, key=lambda a: sum(value(a) for _ in range(simulation)) / simulation)
import numpy as np
from tqdm import tqdm
import pickle

from game import Game2048Env
from ntuple_approximator import NTupleApproximator
from ntuple_action import get_best_action


env = Game2048Env()  # Initialize the game environment
with open('n_tuple_approximator.pkl', 'rb') as f:
    approximator: NTupleApproximator = pickle.load(f)

NUM_GAMES = 10
scores = []

for i in tqdm(range(NUM_GAMES)):
    state = env.reset()
    env.render()
    done = False

    while not done:
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        if not legal_moves:
            break

        best_action = get_best_action(env, approximator, legal_moves)

        action = best_action  # Choose the best action based on evaluation
        state, reward, done, _ = env.step(action)  # Apply the selected action

    scores.append(env.score)

print("Average score over {} games: {:.2f}".format(NUM_GAMES, np.mean(scores)))
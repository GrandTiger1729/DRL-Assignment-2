from typing import List
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing

from game2048env import Game2048Env
from ntuple_approximator import NTupleApproximator
from ntuple_action import get_td_best_action

def generate_trajectory(env: Game2048Env, approximator: NTupleApproximator):
    """
    Generates a single trajectory for an episode.
    """
    state = env.board.copy()
    trajectory = []
    done = False
    max_tile: int = np.max(state)

    while not done:
        legal_moves = env.get_legal_moves()
        action = get_td_best_action(env, approximator, legal_moves)
        next_state, reward, done = env.step(action)
        max_tile = max(max_tile, np.max(next_state))
        trajectory.append((state, reward, next_state, max_tile))

        state = next_state
        

    return trajectory, max_tile, env.score

def td_learning_multiprocessing(env, approximator: NTupleApproximator, num_episodes=50000, alpha=0.01, batch_size=100, workers=4, checkpoint_interval=1000):

    final_scores = []
    max_tiles = []

    GameEnv = env.__class__

    global worker
    def worker(_):
        return generate_trajectory(GameEnv(), approximator)

    for batch_start in tqdm(range(0, num_episodes, batch_size)):
        with multiprocessing.Pool(workers) as pool:
            trajectories = pool.map(worker, range(batch_size))

        for trajectory, max_tile, final_score in trajectories:

            for state, reward, next_state, max_t in reversed(trajectory):

                delta = reward + approximator.get_value(next_state) + approximator.get_value(state)
                approximator.update(state, delta, alpha)

            final_scores.append(final_score)
            max_tiles.append(max_tile)

        if (batch_start + batch_size) % checkpoint_interval == 0:
            avg_score = np.mean(final_scores[-checkpoint_interval:])
            avg_max_tile = np.mean(max_tiles[-checkpoint_interval:])
            success_rate = np.mean([1 if max_tile >= 4096 else 0 for max_tile in max_tiles[-checkpoint_interval:]])
            # state_count = sum(len(weight) for weight in approximator.weights)
            
            print(f"Episode {batch_start + batch_size}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Avg Max Tile: {avg_max_tile:.2f}")
            approximator.save()

    return final_scores

# patterns = [
#     [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
#     [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
#     [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
#     [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
#     [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
#     [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
#     [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
#     [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)],
# ]

env = Game2048Env()
approximator = NTupleApproximator()
approximator.load()

td_learning_multiprocessing(env, approximator, num_episodes=80000, alpha=0.08, batch_size=40, workers=16, checkpoint_interval=400)

# Save the approximator
approximator.save()
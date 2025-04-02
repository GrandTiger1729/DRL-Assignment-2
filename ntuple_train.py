import numpy as np
from tqdm import tqdm
import pickle

from game import Game2048Env
from ntuple_approximator import NTupleApproximator
from ntuple_action import get_best_action


def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """

    final_scores = []
    max_tiles = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            action = get_best_action(env, approximator, legal_moves, gamma)

            next_state, new_score, done, _ = env.step(action)
            if done:
                new_score -= 10000
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TODO: Store trajectory or just update depending on the implementation
            trajectory.append((state, action, incremental_reward, next_state))

            state = next_state

        # # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
        for state, action, incremental_reward, next_state in reversed(trajectory):
            delta = incremental_reward + gamma * approximator.value(next_state) - approximator.value(state)
            approximator.update(state, delta, alpha)

        final_scores.append(env.score)
        max_tiles.append(max_tile)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            avg_max_tile = np.mean(max_tiles[-100:])
            success_rate = np.mean([1 if max_tile >= 2048 else 0 for max_tile in max_tiles[-100:]])
            state_count = sum(len(weight) for weight in approximator.weights)
            
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Avg Max Tile: {avg_max_tile:.2f} | State Count: {state_count}")
            with open('ntuple_approximator.pkl', 'wb') as f:
                pickle.dump(approximator, f)

    return final_scores

# TODO: Define your own n-tuple patterns
patterns = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
]

env = Game2048Env()
approximator = NTupleApproximator(board_size=4, patterns=patterns)

final_scores = td_learning(env, approximator, num_episodes=1000, alpha=0.08, gamma=1, epsilon=0.1)
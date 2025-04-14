import numpy as np
from tqdm import tqdm
from game2048env import Game2048Env
from ntuple_approximator import NTupleApproximator

def test(num_games, approximator):
    env = Game2048Env()
    avg_score = 0
    avg_max_tile = 0

    for game in tqdm(range(num_games)):
        # Reset environment
        state = env.reset()
        while True:
            legal_moves = env.get_legal_moves()
            assert legal_moves, "No legal moves available."

            # Choose the best action using TD(0) greedy
            values = []
            for action in legal_moves:
                temp_env = Game2048Env()
                temp_env.board = env.board.copy()
                temp_env.score = env.score
                next_state, reward, _done = temp_env.step(action)
                values.append(reward + approximator.get_value(next_state))

            
            best_action = legal_moves[np.argmax(values)]

            # Perform the action
            next_state, reward, done = env.step(best_action)
            env.render()
            # Update state
            state = next_state
            if done:
                break

        # Statistics
        avg_score += env.score
        mx = np.max(env.board)
        avg_max_tile += mx

    # Print results
    print(
        f"Avg Score: {avg_score / num_games:.2f}, "
        f"Avg Max Tile: {avg_max_tile / num_games:.2f}"
    )

NUM_GAMES = 10
if __name__ == "__main__":
    env = Game2048Env()  # Initialize the game environment
    approximator = NTupleApproximator()
    approximator.load()
    test(NUM_GAMES, approximator)
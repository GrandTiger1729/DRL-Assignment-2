# Remember to adjust your student ID in meta.xml
import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import random
import copy


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.step_count = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.cache_after_states = [None] * 4 # Cache for after states of each action, (board, score, is legal)

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.cache_after_states = [None] * 4
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy()

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        previous_score = self.score
        # previous_score = self.evaluate()

        if self.cache_after_states[action] is None:
            self.simulate_move(action)
        
        new_board, new_score, moved = self.cache_after_states[action]
        if moved:
            self.board = new_board
            self.score = new_score
            self.cache_after_states = [None] * 4  # Clear cache after a valid move
            self.add_random_tile()
            self.step_count += 1
        
        done = self.is_game_over()

        return self.board.copy(), self.score - previous_score, done
        # return self.board.copy(), self.evaluate() - previous_score, done

    def evaluate(self):
        adj_score = [0.0] * 4

        # Horizontal smoothness and monotonicity
        for i in range(4):
            j = 0
            while j < 4 and self.board[i][j] == 0:
                j += 1
            if j < 4:
                k = j + 1
                while k < 4:
                    while k < 4 and self.board[i][k] == 0:
                        k += 1
                    if k == 4:
                        break
                    if self.board[i][j] < self.board[i][k]:
                        adj_score[0] += np.log2(self.board[i][k]) - np.log2(self.board[i][j])
                    elif self.board[i][j] > self.board[i][k]:
                        adj_score[1] += np.log2(self.board[i][j]) - np.log2(self.board[i][k])
                    j = k
                    k += 1

        # Vertical smoothness and monotonicity
        for j in range(4):
            i = 0
            while i < 4 and self.board[i][j] == 0:
                i += 1
            if i < 4:
                k = i + 1
                while k < 4:
                    while k < 4 and self.board[k][j] == 0:
                        k += 1
                    if k == 4:
                        break
                    if self.board[i][j] < self.board[k][j]:
                        adj_score[2] += np.log2(self.board[k][j]) - np.log2(self.board[i][j])
                    elif self.board[i][j] > self.board[k][j]:
                        adj_score[3] += np.log2(self.board[i][j]) - np.log2(self.board[k][j])
                    i = k
                    k += 1

        smoothness = sum(adj_score)
        mono = max(adj_score[0], adj_score[1]) + max(adj_score[2], adj_score[3])

        # Count empty cells
        empty_cells = sum(1 for i in range(4) for j in range(4) if self.board[i][j] == 0)

        # Check if the game is over
        done = self.is_game_over()

        # Calculate evaluation score
        eval_score = (
            self.score
            - 0.1 * smoothness
            + 1 * mono
            + 2.7 * empty_cells
            - 1e5 * done
        )
        return eval_score

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        reward = 0
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
                reward += new_row[i]
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row, reward
    
    def simulate_move(self, action):
        """Simulate a move without changing the board"""
        
        # Return the cached state if it exists
        if self.cache_after_states[action] is not None:    
            return self.cache_after_states[action]
        
        # Create a copy of the current board state
        temp_board = self.board.copy()
        score = self.score

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col, reward = self.simulate_row_move(col)
                temp_board[:, j] = new_col
                score += reward
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col, reward = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
                score += reward
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i], reward = self.simulate_row_move(row)
                score += reward
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row, reward = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
                score += reward
        else:
            raise ValueError("Invalid action")
        
        moved = not np.array_equal(self.board, temp_board)
        # Cache the after state for the action
        self.cache_after_states[action] = (temp_board, score, moved)
        
        return self.cache_after_states[action]

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""

        if self.cache_after_states[action] is None:
            self.simulate_move(action)

        temp_board, temp_score, moved = self.cache_after_states[action]
        # If the simulated board is different from the current board, the move is legal
        return moved
    
    def get_legal_moves(self):
        """Get all legal moves"""
        return [a for a in range(4) if self.is_move_legal(a)]
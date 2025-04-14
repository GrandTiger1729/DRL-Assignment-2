from typing import Optional, Tuple, List, Dict, Any
import sys
import numpy as np
import random
from collections import defaultdict

def evaluate_board(board: np.ndarray, color: int) -> float:
    n = board.shape[0]
    # Check instant win
    counts = [[0] * 7, [0] * 7]
    for i in range(n):
        for j in range(n):
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                color_count = [0, 0, 0]
                for k in range(6, 0, -1):
                    ni = i + di * k
                    nj = j + dj * k
                    if not (0 <= ni < n and 0 <= nj < n):
                        break
                    color_count[board[ni, nj]] += 1
                    if color_count[1] > 0 and color_count[2] > 0:
                        break
                else:
                    if color_count[1] > 0 and color_count[2] == 0:
                        counts[0][color_count[1]] += 1
                    if color_count[2] > 0 and color_count[1] == 0:
                        counts[1][color_count[2]] += 1

    if color == 1:
        if counts[0][4] > 0 or counts[0][5] > 0 or counts[0][6] > 0:
            return 1.0
        if counts[1][5] > 0 or counts[1][6] > 0:
            return -1.0
        
        me = np.argmax(counts[0][1:])
        opponent = np.argmax(counts[1][1:])

        return (me / 6) ** 3 - (opponent / 6)
    else:
        if counts[1][4] > 0 or counts[1][5] > 0 or counts[1][6] > 0:
            return -1.0
        if counts[0][5] > 0 or counts[0][6] > 0:
            return 1.0
        
        me = np.argmax(counts[1][1:])
        opponent = np.argmax(counts[0][1:])

        return (me / 6) ** 3 - (opponent / 6)


def recommended_moves(board: np.ndarray, color: int, size=5) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Generates recommended moves for the given board and color."""
    n = board.shape[0]
    # Check instant win
    for i in range(n):
        for j in range(n):
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                empty_cells = []
                for k in range(6, 0, -1):
                    ni = i + di * k
                    nj = j + dj * k
                    if not (0 <= ni < n and 0 <= nj < n):
                        break
                    if board[ni, nj] == 3 - color:
                        break
                    if board[ni, nj] == 0:
                        empty_cells.append((ni, nj))
                else:
                    if len(empty_cells) == 1:
                        for _i in range(n):
                            for _j in range(n):
                                if board[_i, _j] == 0 and (_i, _j) != empty_cells[0]:
                                    return [(empty_cells[0], (_i, _j))]
                    if len(empty_cells) == 2:
                        return [tuple(empty_cells)]
    
    candidate_moves = []
    for i1 in range(n):
        for j1 in range(n):
            if board[i1, j1] != 0:
                continue
            for i2 in range(n):
                for j2 in range(n):
                    if board[i2, j2] != 0  or i1 * n + j1 >= i2 * n + j2:
                        continue
                    temp_board = board.copy()
                    temp_board[i1, j1] = color
                    temp_board[i2, j2] = color
                    candidate_moves.append(((i1, j1), (i2, j2), 1 - evaluate_board(temp_board, color)))

    candidate_moves.sort(key=lambda x: x[2], reverse=True)
    recommended_moves = []
    for i in range(min(size, len(candidate_moves))):
        recommended_moves.append(((candidate_moves[i][0]), (candidate_moves[i][1])))
    return recommended_moves

class MCTSNode:
    def __init__(self, board: np.ndarray, color, parent=None):
        self.board = board.copy()
        self.color = color

        self.parent: Optional[MCTSNode] = parent
        self.children: Dict[List[Tuple[int, int]], MCTSNode] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = recommended_moves(board, color)

    def fully_expanded(self):
        return len(self.untried_actions) == 0
    

class MCTS:
    def __init__(self, iterations=50, exploration_constant=1.41):
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation

    def select_child(self, node: MCTSNode) -> MCTSNode:
        def value(child: MCTSNode):
            assert child.visits > 0, "Child node has no visits"
            return child.total_reward / child.visits + self.c * np.sqrt(np.log(node.visits) / child.visits)
        
        return max(node.children.values(), key=value)

    def expand(self, node: MCTSNode) -> MCTSNode:
        action = node.untried_actions.pop()
        board = node.board.copy()
        color = node.color
        board[action[0]] = color
        board[action[1]] = color
        child_node = MCTSNode(board, 3 - color, parent=node)
        node.children[action] = child_node
        return child_node
    
    def rollout(self, board: np.ndarray, color: int) -> float:
        return evaluate_board(board, color)
    
    def backpropagate(self, node: MCTSNode, total_reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += total_reward
            total_reward = -total_reward  # Invert reward for the opponent
            node = node.parent

    def terminate(self, board: np.ndarray) -> bool:
        """Check connect 6."""
        n = board.shape[0]
        for i in range(n):
            for j in range(n):
                if board[i, j] != 0:
                    color = board[i, j]
                    for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                        count = 0
                        for k in range(6):
                            ni = i + di * k
                            nj = j + dj * k
                            if not (0 <= ni < n and 0 <= nj < n):
                                break
                            if board[ni, nj] == color:
                                count += 1
                            else:
                                break
                        if count >= 6:
                            return True
        return False

    def simulate(self, root: MCTSNode):
        node = root
        while not self.terminate(node.board) and node.fully_expanded():
            node = self.select_child(node)
            
        if not node.fully_expanded():
            node = self.expand(node)

        reward = self.rollout(node.board, node.color)
        self.backpropagate(node, reward)
        

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        if self.board.sum() == 0:
            move_str = "J10"
            self.play_move(color, move_str)
            print(move_str, file=sys.stderr)
            return
        
        root = MCTSNode(self.board, self.turn)
        mcts = MCTS()
        for _ in range(mcts.iterations):
            mcts.simulate(root)

        best_move = max(root.children.items(), key=lambda x: x[1].visits)
        selected = best_move[0]
        move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                raise e
                # print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()

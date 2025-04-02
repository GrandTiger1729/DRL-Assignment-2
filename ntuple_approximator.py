import copy
import math
from collections import defaultdict

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for group, pattern in enumerate(self.patterns):
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append((syms_, group))

    def rot90(self, coord):
        row, col = coord
        return col, self.board_size - 1 - row

    def reflect(self, coord):
        row, col = coord
        return self.board_size - 1 - row, col

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        patterns = []
        pattern = copy.deepcopy(pattern)
        for _ in range(4):
            for __ in range(2):
                patterns.append(tuple(pattern))
                pattern = [self.reflect(coord) for coord in pattern]
            pattern = [self.rot90(coord) for coord in pattern]
        return patterns

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple([self.tile_to_index(board[i, j]) for i, j in coords])

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total = 0
        for pattern, group in self.symmetry_patterns:
            feature = self.get_feature(board, pattern)
            total += self.weights[group][feature]
        return total

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for pattern, group in self.symmetry_patterns:
            feature = self.get_feature(board, pattern)
            self.weights[group][feature] += alpha * delta / len(self.symmetry_patterns)


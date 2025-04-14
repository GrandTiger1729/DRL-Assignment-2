import numpy as np

PATTERNS = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)],
]

class NTupleApproximator:
    def __init__(self, patterns=PATTERNS):
        self.patterns = patterns
        self.symmetry_patterns = [[] for _ in range(len(patterns))]
        self.weights = []

        for k in range(len(patterns)):
            # Initialize weights
            self.weights.append([4e4] * (15 ** len(patterns[k])))  # or use appropriate initialization
            
            # Generate symmetry patterns
            symmetry_pattern = patterns[k]
            for I in range(2):
                for J in range(4):
                    self.symmetry_patterns[k].append(symmetry_pattern.copy())
                    # rotate 90 degrees
                    for i in range(len(symmetry_pattern)):
                        x, y = symmetry_pattern[i]
                        symmetry_pattern[i] = (y, 3 - x)
                # reflect over y-axis
                for i in range(len(symmetry_pattern)):
                    symmetry_pattern[i] = (3 - symmetry_pattern[i][0], symmetry_pattern[i][1])

    def get_feature(self, grid, pattern):
        feature = []
        for x, y in pattern:
            feature.append(int(np.log2(grid[x][y])) if grid[x][y] != 0 else 0)
        return feature

    def get_index(self, feature):
        idx = 0
        pw = 1
        for f in feature:
            idx += f * pw
            pw *= 15
        return idx

    def get_value(self, grid):
        value = 0
        for k in range(len(self.patterns)):
            for pattern in self.symmetry_patterns[k]:
                feature = self.get_feature(grid, pattern)
                idx = self.get_index(feature)
                value += self.weights[k][idx]

        return value

    def update(self, grid, delta, alpha):
        for k in range(len(self.patterns)):
            for pattern in self.symmetry_patterns[k]:
                feature = self.get_feature(grid, pattern)
                idx = self.get_index(feature)
                self.weights[k][idx] += alpha * delta / len(self.patterns) / 8

    def save(self, filename="weights.txt"):
        with open(filename, 'w') as file:
            for k in range(len(self.patterns)):
                # Join weights for each pattern as space-separated values
                file.write(' '.join(map(str, self.weights[k])) + '\n')

    def load(self, filename="weights.txt"):
        with open(filename, 'r') as file:
            for k in range(len(self.patterns)):
                # Read weights from the file for each pattern
                self.weights[k] = list(map(float, file.readline().strip().split()))
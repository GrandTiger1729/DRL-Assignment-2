#include <bits/stdc++.h>
using namespace std;

typedef vector<vector<int>> Grid;
typedef vector<pair<int, int>> Pattern;

struct Game2048Env {
  Grid grid;  // in log space, 0 means empty, 1 means 2, 2 means 4, etc.
  int score;

  Game2048Env() { reset(); }
  void render() {
    cout << "Score: " << score << endl;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        if (grid[i][j] == 0) {
          cout << setw(6) << " " << ",";
        } else {
          cout << setw(6) << (1 << grid[i][j]) << ",";
        }
      }
      cout << endl;
    }
  }
  Grid reset() {
    grid = vector<vector<int>>(4, vector<int>(4, 0));
    score = 0;
    add_random_tile();
    add_random_tile();
    return grid;
  }
  void add_random_tile() {
    vector<pair<int, int>> empty_tiles;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        if (grid[i][j] == 0) {
          empty_tiles.push_back({i, j});
        }
      }
    }
    if (!empty_tiles.empty()) {
      mt19937 rng(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
      auto [x, y] = empty_tiles[rng() % empty_tiles.size()];
      grid[x][y] = (rng() % 10 == 0) ? 2 : 1;  // 10% chance of 4
    }
  }
  bool move_up() {
    bool moved = false;
    for (int j = 0; j < 4; j++) {
      int last = 0;
      for (int i = 1; i < 4; i++) {
        if (grid[i][j] == 0) {
          continue;
        }
        assert(last != i);
        if (grid[last][j] == 0) {
          swap(grid[last][j], grid[i][j]);
          moved = true;
        } else if (grid[i][j] == grid[last][j]) {
          grid[last][j] += 1;
          score += (1 << grid[last][j]);
          grid[i][j] = 0;
          last += 1;
          moved = true;
        } else {
          last += 1;
          assert(0 <= last && last < 4);
          if (last != i) {
            swap(grid[last][j], grid[i][j]);
            moved = true;
          }
        }
      }
    }
    return moved;
  }
  bool move_down() {
    bool moved = false;
    for (int j = 0; j < 4; j++) {
      int last = 3;
      for (int i = 2; i >= 0; i--) {
        if (grid[i][j] == 0) {
          continue;
        }
        assert(last != i);
        if (grid[last][j] == 0) {
          swap(grid[last][j], grid[i][j]);
          moved = true;
        } else if (grid[i][j] == grid[last][j]) {
          grid[last][j] += 1;
          score += (1 << grid[last][j]);
          grid[i][j] = 0;
          last -= 1;
          moved = true;
        } else {
          last -= 1;
          assert(0 <= last && last < 4);
          if (last != i) {
            swap(grid[last][j], grid[i][j]);
            moved = true;
          }
        }
      }
    }
    return moved;
  }
  bool move_left() {
    bool moved = false;
    for (int i = 0; i < 4; i++) {
      int last = 0;
      for (int j = 1; j < 4; j++) {
        if (grid[i][j] == 0) {
          continue;
        }
        assert(last != j);
        if (grid[i][last] == 0) {
          swap(grid[i][last], grid[i][j]);
          moved = true;
        } else if (grid[i][j] == grid[i][last]) {
          grid[i][last] += 1;
          score += (1 << grid[i][last]);
          grid[i][j] = 0;
          last += 1;
          moved = true;
        } else {
          last += 1;
          assert(0 <= last && last < 4);
          if (last != j) {
            swap(grid[i][last], grid[i][j]);
            moved = true;
          }
        }
      }
    }
    return moved;
  }
  bool move_right() {
    bool moved = false;
    for (int i = 0; i < 4; i++) {
      int last = 3;
      for (int j = 2; j >= 0; j--) {
        if (grid[i][j] == 0) {
          continue;
        }
        assert(last != j);
        if (grid[i][last] == 0) {
          swap(grid[i][last], grid[i][j]);
          moved = true;
        } else if (grid[i][j] == grid[i][last]) {
          grid[i][last] += 1;
          score += (1 << grid[i][last]);
          grid[i][j] = 0;
          last -= 1;
          moved = true;
        } else {
          last -= 1;
          assert(0 <= last && last < 4);
          if (last != j) {
            swap(grid[i][last], grid[i][j]);
            moved = true;
          }
        }
      }
    }
    return moved;
  }
  tuple<Grid, int, bool> step(int action) {
    int previous_score = score;

    bool moved;
    if (action == 0) {
      moved = move_up();
    } else if (action == 1) {
      moved = move_down();
    } else if (action == 2) {
      moved = move_left();
    } else if (action == 3) {
      moved = move_right();
    } else {
      assert(false);
    }
    if (moved) {
      add_random_tile();
    }

    int reward = score - previous_score;
    bool done = is_game_over();
    return make_tuple(grid, reward, done);
  }
  bool is_game_over() { return get_legal_moves().empty(); }
  bool is_legal_move(int action) {
    if (action == 0) {
      for (int j = 0; j < 4; j++) {
        for (int i = 1; i < 4; i++) {
          if (grid[i][j] == 0) {
            continue;
          }
          if (grid[i - 1][j] == 0 || grid[i][j] == grid[i - 1][j]) {
            return true;
          }
        }
      }
    } else if (action == 1) {
      for (int j = 0; j < 4; j++) {
        for (int i = 2; i >= 0; i--) {
          if (grid[i][j] == 0) {
            continue;
          }
          if (grid[i + 1][j] == 0 || grid[i][j] == grid[i + 1][j]) {
            return true;
          }
        }
      }
    } else if (action == 2) {
      for (int i = 0; i < 4; i++) {
        for (int j = 1; j < 4; j++) {
          if (grid[i][j] == 0) {
            continue;
          }
          if (grid[i][j - 1] == 0 || grid[i][j] == grid[i][j - 1]) {
            return true;
          }
        }
      }
    } else if (action == 3) {
      for (int i = 0; i < 4; i++) {
        for (int j = 2; j >= 0; j--) {
          if (grid[i][j] == 0) {
            continue;
          }
          if (grid[i][j + 1] == 0 || grid[i][j] == grid[i][j + 1]) {
            return true;
          }
        }
      }
    } else {
      assert(false);
    }
    return false;
  }
  vector<int> get_legal_moves() {
    vector<int> legal_moves;
    for (int action = 0; action < 4; action++) {
      if (is_legal_move(action)) {
        legal_moves.push_back(action);
      }
    }
    return legal_moves;
  }
  float evaluate() {
    vector<float> adj_score(4, 0);
    for (int i = 0; i < 4; i++) {
      int j = 0;
      while (j < 4 && grid[i][j] == 0) {
        j++;
      }
      if (j < 4) {
        int k = j + 1;
        while (k < 4) {
          while (k < 4 && grid[i][k] == 0) {
            k++;
          }
          if (k == 4) {
            break;
          }
          if (grid[i][j] < grid[i][k]) {
            adj_score[0] += grid[i][k] - grid[i][j];
          } else if (grid[i][j] > grid[i][k]) {
            adj_score[1] += grid[i][j] - grid[i][k];
          }
          j = k;
          k++;
        }
      }
    }

    for (int j = 0; j < 4; j++) {
      int i = 0;
      while (i < 4 && grid[i][j] == 0) {
        i++;
      }
      if (i < 4) {
        int k = i + 1;
        while (k < 4) {
          while (k < 4 && grid[k][j] == 0) {
            k++;
          }
          if (k == 4) {
            break;
          }
          if (grid[i][j] < grid[k][j]) {
            adj_score[2] += grid[k][j] - grid[i][j];
          } else if (grid[i][j] > grid[k][j]) {
            adj_score[3] += grid[i][j] - grid[k][j];
          }
          i = k;
          k++;
        }
      }
    }

    float smoothness =
        adj_score[0] + adj_score[1] + adj_score[2] + adj_score[3];
    float mono =
        max(adj_score[0], adj_score[1]) + max(adj_score[2], adj_score[3]);
    int empty_cells = 0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < grid[i].size(); j++) {
        if (grid[i][j] == 0) {
          empty_cells++;
        }
      }
    }
    bool done = is_game_over();
    float eval_score =
        score - 0.1 * smoothness + 1 * mono + 2.7 * empty_cells - 1e5 * done;
    return eval_score;
  }
};

void test_env() {
  Game2048Env env;
  env.render();
  int i = 0;
  while (true) {
    mt19937 rng(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    int action = rng() % 4;
    auto [grid, reward, done] = env.step(action);
    cout << "==========" << endl;
    cout << "Step: " << i + 1 << endl;
    env.render();
    cout << "Action: " << action << ", Reward: " << reward << endl;
    if (done) {
      cout << "Game Over!" << endl;
      break;
    }
    i++;
  }
}

struct NTupleApproximator {
  vector<Pattern> patterns;
  vector<vector<Pattern>> symmetry_patterns;
  vector<vector<float>> weights;

  NTupleApproximator(vector<Pattern> patterns)
      : patterns(patterns), symmetry_patterns(patterns.size()) {
    for (int k = 0; k < patterns.size(); k++) {
      weights.push_back(vector<float>(pow(15, patterns[k].size()),
                                      4e4 /* patterns.size() / 8 */));

      // Generate symmetry patterns
      auto symmetry_pattern = patterns[k];
      for (int I = 0; I < 2; I++) {
        for (int J = 0; J < 4; J++) {
          symmetry_patterns[k].push_back(symmetry_pattern);
          // rotate 90 degrees
          for (int i = 0; i < symmetry_pattern.size(); i++) {
            auto [x, y] = symmetry_pattern[i];
            symmetry_pattern[i].first = y;
            symmetry_pattern[i].second = 3 - x;
          }
        }
        // reflect over y-axis
        for (int i = 0; i < symmetry_pattern.size(); i++) {
          symmetry_pattern[i].first = 3 - symmetry_pattern[i].first;
        }
      }
    }
  }
  vector<int> get_feature(const Grid &grid, Pattern pattern) {
    vector<int> feature;
    for (int i = 0; i < pattern.size(); i++) {
      auto [x, y] = pattern[i];
      feature.push_back(grid[x][y]);
    }
    return feature;
  }
  int get_index(const vector<int> &feature) {
    int idx = 0, pw = 1;
    for (int i = 0; i < feature.size(); i++) {
      idx += feature[i] * pw;
      pw *= 15;
    }
    return idx;
  }
  float get_value(const Grid &grid) {
    float value = 0;
    for (int k = 0; k < patterns.size(); k++) {
      for (const auto pattern : symmetry_patterns[k]) {
        auto feature = get_feature(grid, pattern);
        int idx = get_index(feature);
        value += weights[k][idx];
      }
    }
    return value;
  }
  void update(const Grid &grid, float delta, float alpha) {
    for (int k = 0; k < patterns.size(); k++) {
      for (const auto pattern : symmetry_patterns[k]) {
        auto feature = get_feature(grid, pattern);
        int idx = get_index(feature);
        weights[k][idx] += alpha * delta / patterns.size() / 8;
      }
    }
  }
  void save(const string &filename) {
    ofstream file(filename);
    for (int k = 0; k < patterns.size(); k++) {
      for (int i = 0; i < weights[k].size(); i++) {
        file << weights[k][i] << " ";
      }
      file << endl;
    }
    file.close();
  }
  void load(const string &filename) {
    ifstream file(filename);
    for (int k = 0; k < patterns.size(); k++) {
      for (int i = 0; i < weights[k].size(); i++) {
        file >> weights[k][i];
      }
    }
    file.close();
  }
};

// class ThreadPool {
//  public:
//   ThreadPool(size_t num_threads) {
//     for (size_t i = 0; i < num_threads; ++i) {
//       workers.emplace_back([this] {
//         while (true) {
//           std::function<void()> task;
//           {
//             std::unique_lock<std::mutex> lock(this->queue_mutex);
//             this->condition.wait(
//                 lock, [this] { return this->stop || !this->tasks.empty(); });
//             if (this->stop && this->tasks.empty()) return;
//             task = std::move(this->tasks.front());
//             this->tasks.pop();
//             working_threads++;
//           }
//           task();
//           working_threads--;
//           finished_condition.notify_all();
//         }
//       });
//     }
//   }
//   ~ThreadPool() {
//     {
//       std::unique_lock<std::mutex> lock(queue_mutex);
//       stop = true;
//     }
//     condition.notify_all();
//     for (std::thread &worker : workers) worker.join();
//   }

//   template <class F>
//   void enqueue(F &&f) {
//     {
//       std::unique_lock<std::mutex> lock(queue_mutex);
//       tasks.emplace(std::forward<F>(f));
//     }
//     condition.notify_one();
//   }

//   void wait_for_tasks() {
//     std::unique_lock<std::mutex> lock(queue_mutex);
//     finished_condition.wait(
//         lock, [this] { return tasks.empty() && working_threads.load() == 0; });
//   }

//  private:
//   std::vector<std::thread> workers;
//   std::queue<std::function<void()>> tasks;

//   std::mutex queue_mutex;
//   std::condition_variable condition;
//   bool stop = false;

//   std::condition_variable finished_condition;
//   std::atomic<int> working_threads{0};
// };

void train(int num_episodes, NTupleApproximator &approximator,
           float alpha = 0.1) {
  Game2048Env env;
  double avg_score = 0;     // last 100 episodes
  double avg_max_tile = 0;  // in log scale, last 100 episodes
  double start_time = clock();

  for (int episode = 0; episode < num_episodes; episode++) {
    vector<tuple<Grid, int, Grid>> trajectory;  // state, reward, next_state

    // Reset environment
    Grid state = env.reset();
    float previous_eval = env.evaluate();
    while (true) {
      auto legal_moves = env.get_legal_moves();
      assert(!legal_moves.empty());

      // Choose best action by td(0) greedy
      vector<float> values(legal_moves.size());
      transform(legal_moves.begin(), legal_moves.end(), values.begin(),
                [&](int action) {
                  auto temp_env = env;
                  auto [grid, reward, done] = temp_env.step(action);
                  return reward + approximator.get_value(grid);
                });
      int action = legal_moves[max_element(values.begin(), values.end()) -
                               values.begin()];

      // float previous_eval = env.evaluate();
      auto [grid, reward, done] = env.step(action);
      float current_eval = env.evaluate();
      float shaped_reward = reward + current_eval - previous_eval;
      previous_eval = current_eval;

      trajectory.push_back({state, shaped_reward, grid});

      // Update state
      state = grid;
      if (done) {
        break;
      }
    }

    // Update weights
    for (int i = trajectory.size() - 1; i >= 0; i--) {
      auto [state, reward, next_state] = trajectory[i];
      float value = approximator.get_value(state);
      float next_value = approximator.get_value(next_state);
      float delta = reward + next_value - value;
      approximator.update(state, delta, alpha);
    }

    // Statistics
    avg_score += env.score;
    int mx = 0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        mx = max(mx, env.grid[i][j]);
      }
    }
    avg_max_tile += mx;

    // Print progress
    if ((episode + 1) % 100 == 0) {
      double elapsed_time = (clock() - start_time) / CLOCKS_PER_SEC;

      cout << fixed << "Episode:" << setw(6) << episode + 1
           << ", Score:" << setw(10) << setprecision(2) << avg_score / 100
           << ", Max Tile:" << setw(10) << setprecision(2)
           << pow(2, avg_max_tile / 100) << ", Time:" << setw(10)
           << setprecision(2) << elapsed_time << endl;

      avg_score = 0;
      avg_max_tile = 0;
      start_time = clock();
    }
    // Save weights every 1000 episodes
    if ((episode + 1) % 5000 == 0) {
      cout << "Saving weights..." << endl;
      approximator.save("weights.txt");
      cout << "Weights saved." << endl;
    }
  }
}
// void train_parallel_batch(
//     int num_episodes, NTupleApproximator &approximator, float alpha = 0.1,
//     int batch_size = 16,
//     int num_threads = std::thread::hardware_concurrency()) {
//   std::mutex update_mutex;
//   std::vector<std::tuple<Grid, float>> batched_updates;

//   std::atomic<int> episodes_done = 0;
//   double avg_score = 0, avg_max_tile = 0;
//   double start_time = clock();

//   ThreadPool pool(num_threads);

//   auto simulate_episode = [&](int id) {
//     Game2048Env env;
//     vector<std::tuple<Grid, float, Grid>> trajectory;

//     Grid state = env.reset();
//     float previous_eval = env.evaluate();

//     while (true) {
//       auto legal_moves = env.get_legal_moves();
//       vector<float> values(legal_moves.size());

//       std::transform(legal_moves.begin(), legal_moves.end(), values.begin(),
//                      [&](int action) {
//                        auto temp_env = env;
//                        auto [grid, reward, done] = temp_env.step(action);
//                        return reward + approximator.get_value(grid);
//                      });

//       int action = legal_moves[max_element(values.begin(), values.end()) -
//                                values.begin()];
//       auto [grid, reward, done] = env.step(action);
//       float current_eval = env.evaluate();
//       float shaped_reward = reward + current_eval - previous_eval;
//       previous_eval = current_eval;

//       trajectory.push_back({state, shaped_reward, grid});
//       state = grid;

//       if (done) break;
//     }

//     // Compute deltas and accumulate
//     std::vector<std::tuple<Grid, float>> local_updates;
//     for (int i = trajectory.size() - 1; i >= 0; i--) {
//       auto [state, reward, next_state] = trajectory[i];
//       float value = approximator.get_value(state);
//       float next_value = approximator.get_value(next_state);
//       float delta = reward + next_value - value;
//       local_updates.push_back({state, delta});
//     }

//     {
//       std::lock_guard<std::mutex> lock(update_mutex);
//       for (auto &upd : local_updates) batched_updates.push_back(upd);

//       avg_score += env.score;
//       int mx = 0;
//       for (int i = 0; i < 4; i++)
//         for (int j = 0; j < 4; j++) mx = max(mx, env.grid[i][j]);
//       avg_max_tile += mx;

//       episodes_done++;

//       // If enough updates collected, apply batch
//       if (batched_updates.size() >= batch_size) {
//         for (auto &[state, delta] : batched_updates) {
//           approximator.update(state, delta, alpha);
//         }
//         batched_updates.clear();
//       }

//       if (episodes_done % 100 == 0) {
//         double elapsed_time = (clock() - start_time) / CLOCKS_PER_SEC;
//         cout << fixed << "Episode:" << setw(6) << episodes_done
//              << ", Score:" << setw(10) << setprecision(2) << avg_score / 100
//              << ", Max Tile:" << setw(10) << setprecision(2)
//              << pow(2, avg_max_tile / 100) << ", Time:" << setw(10)
//              << setprecision(2) << elapsed_time << endl;
//         avg_score = 0;
//         avg_max_tile = 0;
//         start_time = clock();
//       }

//       if (episodes_done % 5000 == 0) {
//         cout << "Saving weights..." << endl;
//         approximator.save("weights.txt");
//         cout << "Weights saved." << endl;
//       }
//     }
//   };

//   for (int i = 0; i < num_episodes; ++i) {
//     pool.enqueue([=] { simulate_episode(i); });
//   }

//   pool.wait_for_tasks();

//   // Final flush
//   for (auto &[state, delta] : batched_updates) {
//     approximator.update(state, delta, alpha);
//   }
//   if (!batched_updates.empty()) {
//     cout << "Final batch of updates applied." << endl;
//     cout << "Saving weights..." << endl;
//     approximator.save("weights.txt");
//     cout << "Weights saved." << endl;
//   }
// }

void test(int num_games, NTupleApproximator &approximator) {
  Game2048Env env;
  double avg_score = 0;
  double avg_max_tile = 0;

  for (int game = 0; game < num_games; game++) {
    // Reset environment
    Grid state = env.reset();
    while (true) {
      auto legal_moves = env.get_legal_moves();
      assert(!legal_moves.empty());

      // Choose best action by td(0) greedy
      vector<float> values(legal_moves.size());
      transform(legal_moves.begin(), legal_moves.end(), values.begin(),
                [&](int action) {
                  auto temp_env = env;
                  auto [grid, reward, done] = temp_env.step(action);
                  return reward + approximator.get_value(grid);
                });
      int action = legal_moves[max_element(values.begin(), values.end()) -
                               values.begin()];

      auto [grid, reward, done] = env.step(action);

      // Update state
      state = grid;
      if (done) {
        break;
      }
    }

    // Statistics
    avg_score += env.score;
    int mx = 0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        mx = max(mx, env.grid[i][j]);
      }
    }
    avg_max_tile += mx;
  }
  // Print results
  cout << fixed << "Avg Score:" << setw(10) << setprecision(2)
       << avg_score / 100 << ", Avg Max Tile:" << setw(10) << setprecision(2)
       << pow(2, avg_max_tile / 100) << endl;

  avg_score = 0;
  avg_max_tile = 0;
}

int main() {
  // Define patterns
  vector<Pattern> patterns = {
      {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}},
      {{0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 1}, {3, 1}},
      {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}, {1, 1}},
      {{0, 0}, {0, 1}, {1, 1}, {1, 2}, {1, 3}, {2, 2}},
      {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {2, 1}, {2, 2}},
      {{0, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}, {3, 2}},
      {{0, 0}, {0, 1}, {1, 1}, {2, 0}, {2, 1}, {3, 1}},
      {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 2}},
  };
  auto approximator = NTupleApproximator(patterns);
  approximator.load("weights.txt");
  train(300000, approximator, 0.1);
  // test(1, approximator);
  return 0;
}
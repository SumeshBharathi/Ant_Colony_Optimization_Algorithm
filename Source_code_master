import numpy as np
import timeit
import matplotlib.pyplot as plt

class AntColonyGeneralized:
    def __init__(self, problem_type, distances, n_ants, n_best, n_best_update, decay, alpha=1, beta=2):
        self.problem_type = problem_type
        self.distances = distances
        self.pheromone = np.ones_like(distances) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_best_update = n_best_update
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distances[path[i]][path[i + 1]]
        return total_dist

    def gen_all_paths(self):
        paths = []
        for i in self.all_inds:
            all_inds_copy = self.all_inds[:]
            all_inds_copy.remove(i)
            for path in self.gen_paths(i, all_inds_copy):
                paths.append([i] + path)
        return paths

    def gen_paths(self, current_ind, all_inds):
        if len(all_inds) == 0:
            return [[]]
        paths = []
        for i in all_inds:
            inds_left = all_inds[:]
            inds_left.remove(i)
            for path in self.gen_paths(i, inds_left):
                paths.append([i] + path)
        return paths

    def update_pheromone(self, ant_dist):
        for i, row in enumerate(self.pheromone):
            for j, col in enumerate(row):
                self.pheromone[i][j] *= (1 - self.decay)
                if self.pheromone[i][j] < 0.0001:
                    self.pheromone[i][j] = 0.0001
        for ant, dist in ant_dist.items():
            for i in range(len(ant) - 1):
                self.pheromone[ant[i]][ant[i + 1]] += 1.0 / dist * self.decay

    def run(self, iterations):
        all_time = []
        for i in range(iterations):
            ant_dist = {}
            for ant in range(self.n_ants):
                if self.problem_type == "TSP":
                    path = self.gen_single_path()
                elif self.problem_type == "VRP":
                    path = self.gen_single_vrp_path()
                elif self.problem_type == "IRP":
                    path = self.gen_single_irp_path()
                else:
                    raise ValueError("Invalid problem type")
                ant_dist[tuple(path)] = self.gen_path_dist(path)
            self.update_pheromone(ant_dist)
            if i % self.n_best_update == 0:
                self.best_path = min(ant_dist, key=ant_dist.get)
                self.update_pheromone({self.best_path: ant_dist[self.best_path]})
                all_time.append(ant_dist[self.best_path])
        return all_time

    def gen_single_path(self):
        start_node = np.random.choice(self.all_inds)
        allowed_inds = list(self.all_inds)
        allowed_inds.remove(start_node)
        path = [start_node]

        for move in range(len(self.distances) - 1):
            move_choices = self.pheromone[path[-1]][allowed_inds] ** self.alpha * \
                          (1.0 / (self.distances[path[-1]][allowed_inds] + 1e-10)) ** self.beta

            # Check for NaN or infinite probabilities
            if np.isnan(move_choices.sum()) or np.isinf(move_choices.sum()):
                # If probabilities are invalid, choose the next move uniformly at random
                chosen_ind = np.random.choice(allowed_inds)
            else:
                normed_choices = move_choices / (move_choices.sum() + 1e-10)  # Add a small epsilon
                normed_choices /= normed_choices.sum()  # Ensure probabilities sum to 1
                chosen_ind = np.random.choice(allowed_inds, p=normed_choices)

            path.append(chosen_ind)
            allowed_inds.remove(chosen_ind)

        return path

    def gen_single_vrp_path(self):
        start_node = 0
        allowed_inds = list(self.all_inds)[1:]
        np.random.shuffle(allowed_inds)
        path = [start_node] + allowed_inds + [start_node]
        return path

    def gen_single_irp_path(self):
        start_node = 0
        end_node = np.random.choice(self.all_inds)
        path = [start_node]
        current_node = start_node

        while current_node != end_node:
            move_choices = self.pheromone[current_node] ** self.alpha

            # Check for NaN or infinite probabilities
            if np.isnan(move_choices.sum()) or np.isinf(move_choices.sum()):
                # If probabilities are invalid, choose the next move uniformly at random
                next_node = np.random.choice(self.all_inds)
            else:
                normed_choices = move_choices / (move_choices.sum() + 1e-10)  # Add a small epsilon
                normed_choices /= normed_choices.sum()  # Ensure probabilities sum to 1
                next_node = np.random.choice(self.all_inds, p=normed_choices)

            path.append(next_node)
            current_node = next_node

        return path

# Function to calculate time complexity, average, and standard deviation
def evaluate_performance(problem_type, distances, n_ants, n_best, n_best_update, decay, alpha, beta, iterations, runs=5):
    execution_times = []
    best_distances = []

    for _ in range(runs):
        ant_colony = AntColonyGeneralized(problem_type, distances, n_ants, n_best, n_best_update, decay, alpha, beta)

        # Measure execution time using timeit
        execution_time = timeit.timeit(lambda: ant_colony.run(iterations), number=1)

        # Record execution time and best distance
        execution_times.append(execution_time)
        best_distances.append(ant_colony.gen_path_dist(ant_colony.best_path))

    # Calculate average and standard deviation
    avg_execution_time = np.mean(execution_times)
    std_execution_time = np.std(execution_times)
    avg_best_distance = np.mean(best_distances)
    std_best_distance = np.std(best_distances)

    # Display results
    print(f"Problem Type: {problem_type}")
    print("Average Execution Time:", avg_execution_time)
    print("Standard Deviation of Execution Time:", std_execution_time)
    print("Average Best Distance:", avg_best_distance)
    print("Standard Deviation of Best Distance:", std_best_distance)
    print("\n")

    return avg_execution_time, std_execution_time, avg_best_distance, std_best_distance

def visualize_results(problem_types, tsp_results, vrp_results, irp_results):
    plt.figure(figsize=(18, 6))

    # Plot Execution Time
    plt.subplot(1, 2, 1)
    bar_width = 0.2
    index = np.arange(len(problem_types))

    plt.bar(index - bar_width, tsp_results[0], yerr=tsp_results[1], width=bar_width, label='TSP', color='blue', alpha=0.7, capsize=5)
    plt.bar(index, vrp_results[0], yerr=vrp_results[1], width=bar_width, label='VRP', color='green', alpha=0.7, capsize=5)
    plt.bar(index + bar_width, irp_results[0], yerr=irp_results[1], width=bar_width, label='IRP', color='orange', alpha=0.7, capsize=5)

    plt.title('Average Execution Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(index, problem_types)
    plt.legend()

    # Plot Best Distances
    plt.subplot(1, 2, 2)
    plt.bar(index - bar_width, tsp_results[2], yerr=tsp_results[3], width=bar_width, label='TSP', color='blue', alpha=0.7, capsize=5)
    plt.bar(index, vrp_results[2], yerr=vrp_results[3], width=bar_width, label='VRP', color='green', alpha=0.7, capsize=5)
    plt.bar(index + bar_width, irp_results[2], yerr=irp_results[3], width=bar_width, label='IRP', color='orange', alpha=0.7, capsize=5)

    plt.title('Average Best Distance')
    plt.ylabel('Distance')
    plt.xticks(index, problem_types)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example distance matrix for TSP (replace it with your own)
    tsp_distances = np.array([[0, 2, 3, 0, 0],
                              [2, 0, 0, 4, 5],
                              [3, 0, 0, 0, 0],
                              [0, 4, 0, 0, 1],
                              [0, 5, 0, 1, 0]])

    # Example distance matrix for VRP (replace it with your own)
    vrp_distances = np.array([[0, 5, 8, 6, 4],
                              [5, 0, 2, 8, 7],
                              [8, 2, 0, 7, 1],
                              [6, 8, 7, 0, 3],
                              [4, 7, 1, 3, 0]])

    # Example graph for IRP (replace it with your own)
    irp_graph = np.array([[0, 2, 3, 0, 0],
                          [2, 0, 0, 4, 5],
                          [3, 0, 0, 0, 0],
                          [0, 4, 0, 0, 1],
                          [0, 5, 0, 1, 0]])

    problem_types = ["TSP", "VRP", "IRP"]

    tsp_results = evaluate_performance("TSP", tsp_distances, 5, 2, 5, 0.95, 1.0, 2.0, 100, 5)
    vrp_results = evaluate_performance("VRP", vrp_distances, 5, 2, 5, 0.95, 1.0, 2.0, 100, 5)
    irp_results = evaluate_performance("IRP", irp_graph, 5, 2, 5, 0.95, 1.0, 2.0, 100, 5)

    visualize_results(problem_types, tsp_results, vrp_results, irp_results)

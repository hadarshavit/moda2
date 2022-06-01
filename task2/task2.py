import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy.random import rand
import pandas as pd
from numpy import random
from tqdm import tqdm


all_bests = {}
x0s = []
x1s = []
f1s = []
f2s = []

plot_all_trajectories = False
num_points = 10 if plot_all_trajectories else 1000
for i in tqdm(range(num_points + 1)):
    w1 = i / num_points
    w2 = 1 - i / num_points

    def objective(x0, x1):
        f1 = 0.5 * np.power(x0, 2) * x1
        f2 = np.power(x0, 2) + 3 * x0 * x1
        r1 = 0
        r2 = 250
        return np.maximum(w1 * np.abs(f1 - r1), w2 * np.abs(f2 - r2))

    # black-box optimization software
    def local_hillclimber(objective, bounds, n_iterations, step_size, init):
        # generate an initial point
        best = init
        # evaluate the initial point
        best_eval = objective(best[0], best[1])
        curr, curr_eval = best, best_eval  # current working solution
        scores = list()
        points = list()
        for i in range(n_iterations):  # take a step
            
            # candidate = [curr[0] + rand() * step_size[0] - step_size[0] / 2.0,
                        #  curr[1] + rand() * step_size[1] - step_size[1] / 2.0]
            candidate = [np.random.uniform(max(bounds[0, 0], curr[0] - step_size[0]), min(bounds[0, 1], curr[0] + step_size[0])),
                        random.uniform(max(bounds[1, 0], curr[1] - step_size[1]), min(bounds[1, 1], curr[1] + step_size[1]))]

            points.append(candidate)
            # print('>%d f(%s) = %.5f, %s' % (i, best, best_eval, candidate))
            # evaluate candidate point
            # check for new best solution
            candidate_eval = objective(candidate[0], candidate[1])
            if candidate_eval < best_eval:  # store new best point
                best, best_eval = candidate, candidate_eval
                # keep track of scores
                scores.append(best_eval)
                # current best
                curr = candidate
        return [best, best_eval, points, scores]

    bounds = asarray([[0, 5], [0, 10]])
    step_size = asarray([0.4, 0.4])
    n_iterations = 10000
    init = asarray([np.random.uniform(bounds[0, 0], bounds[0, 1]), np.random.uniform(bounds[1, 0], bounds[1, 1])])

    best, score, points, scores, = local_hillclimber(objective, bounds, n_iterations, step_size, init)
    all_bests[f'{round(w1,2)}_{round(w2,2)}'] = {'f1':best[0], 'f2':best[1], 'score':score}
    x0s.append(best[0])
    x1s.append(best[1])
    f1s.append(0.5 * np.power(best[0], 2) * best[1])
    f2s.append(np.power(best[0], 2) + 3 * best[0] * best[1])

    if plot_all_trajectories:
        n, m = 7, 7
        start = -3
        x_vals = np.arange(0, 5, 0.1)
        y_vals = np.arange(0, 10, 0.1)
        X, Y = np.meshgrid(x_vals, y_vals)

        # print(X.shape)
        # print(Y.shape)

        fig = plt.figure(figsize=(6, 5))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        # Z = 0.1 * np.abs((0.5 * np.power(X, 2) * Y)) +
        # 0.9 * np.abs((np.power(X, 2) + 3 * X * Y) - 250)
        Z = objective(X, Y)
        # print(Z)

        cp = ax.contour(X, Y, Z)
        ax.clabel(cp, inline=True, fontsize=10)
        ax.set_title(f'Contour Plot for w1={round(w1,2)}, w2={round(w2,2)}', fontsize=20)
        ax.set_xlabel('x[0]')
        ax.set_ylabel('x[1]')

        for i in range(n_iterations):
            plt.plot(points[i][0], points[i][1], "o")
        plt.savefig(f'pareto{round(w1,2)}_{round(w2,2)}.png')

plt.clf()
random_x1s = random.rand(1000, 1) * (5.0)
random_x2s = random.rand(1000, 1) * (10.0)
plt.scatter(0.5 * np.power(random_x1s, 2) * random_x2s, np.power(random_x1s, 2) + 3 * random_x1s * random_x2s)
plt.scatter(f1s, f2s)
plt.xlabel('F1')
plt.ylabel('F2')
plt.title('Pareto Front')
plt.savefig('pareto_front2.png')
# print('all', all_bests)

plt.clf()

plt.scatter(x0s, x1s)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Efficient Set')
plt.savefig('eff_set.png')

# print(pd.DataFrame.from_dict(all_bests, orient='index').to_latex())


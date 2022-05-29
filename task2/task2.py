import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy.random import rand

all_bests = []
for i in range(11):
    w1 = i / 10.0
    w2 = 1 - i / 10.0


    def objective(x):
        f1 = 0.5 * np.power(x[0], 2) * x[1]
        f2 = np.power(x[0], 2) + 3 * x[0] * x[1]
        r1 = 0
        r2 = 250
        return np.maximum(w1 * np.abs(f1 - r1), w2 * np.abs(f2 - r2))

    # black-box optimization software

    def local_hillclimber(objective, bounds, n_iterations, step_size, init):
        # generate an initial point
        best = init
        # evaluate the initial point
        best_eval = objective(best)
        curr, curr_eval = best, best_eval  # current working solution
        scores = list()
        points = list()
        for i in range(n_iterations):  # take a step
            candidate = [curr[0] + rand() * step_size[0] - step_size[0] / 2.0,
                         curr[1] + rand() * step_size[1] - step_size[1] / 2.0]
            print(candidate)
            if candidate[0] > bounds[0, 1]:
                candidate[0] = curr[0]
            if candidate[1] > bounds[1, 1]:
                candidate[1] = curr[1]
            points.append(candidate)
            print('>%d f(%s) = %.5f, %s' % (i, best, best_eval, candidate))
            # evaluate candidate point
            # check for new best solution
            candidate_eval = objective(candidate)
            if candidate_eval < best_eval:  # store new best point
                best, best_eval = candidate, candidate_eval
                # keep track of scores
                scores.append(best_eval)
                # current best
                curr = candidate
        return [best, best_eval, points, scores]

    bounds = asarray([[0, 5], [0, 10]])
    step_size = [0.2, 0.2]
    n_iterations = 1000
    init = [2.4, 2.0]

    best, score, points, scores, = local_hillclimber(objective, bounds, n_iterations, step_size, init)
    all_bests.append(best)

    n, m = 7, 7
    start = -3
    x_vals = np.arange(0, 5, 0.1)
    y_vals = np.arange(0, 10, 0.1)
    X, Y = np.meshgrid(x_vals, y_vals)

    print(X.shape)
    print(Y.shape)

    fig = plt.figure(figsize=(6, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    # Z = 0.1 * np.abs((0.5 * np.power(X, 2) * Y)) +
    # 0.9 * np.abs((np.power(X, 2) + 3 * X * Y) - 250)
    Z = objective([X, Y])
    print(Z)

    cp = ax.contour(X, Y, Z)
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title(f'Contour Plot for w1={round(w1,2)}, w2={round(w2,2)}', fontsize=20)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')

    for i in range(n_iterations):
        plt.plot(points[i][0], points[i][1], "o")
    plt.savefig(f'pareto{round(w1,2)}_{round(w2,2)}.png')
print('all', all_bests)

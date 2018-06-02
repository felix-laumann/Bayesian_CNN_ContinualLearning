import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import seaborn as sns
import re
import numpy as np
plt.rc('font', family='serif', size=16)
plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16})

tasks = 2  # tasks continually learnt


def load_data(tasks):
    os.chdir("/home/felix/Dropbox/publications/Bayesian_CNN_continual/results/")

    for task in range(1, tasks):
        with open("diagnostics_{}.txt".format(task), 'r') as file:
            acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
        print(acc)

        train = acc[1::2]
        valid = acc[0::2]

        return np.array(train).astype(np.float32), np.array(valid).astype(np.float32)


f = plt.figure(figsize=(10, 8))
colors = sns.color_palette(n_colors=tasks)

train, valid = load_data(tasks=tasks)

#print(valid)
#print(valid.shape[1])

plt.plot(valid, "--", label=r"Validation, prior: $U(a, b)$", color='maroon')
plt.plot(valid, "--", label=r"Validation, prior: $q(w | \theta_A)$", color='navy')


plt.xlabel("Epochs")
plt.ylabel("Accuracy")
x_ticks = range(len(valid))
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))

f.suptitle("Evaluating continual learning")
plt.legend()

plt.savefig("results_continual.png")

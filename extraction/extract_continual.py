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
    files = [open("diagnostics_{}.txt".format(task)).read() for task in range(tasks)]

    for file in files:
        with open(file, 'r') as f:
            acc = re.findall(r"'acc':\s+tensor\((.*?)\)", f.read())
        print(acc)

        train[file] = acc[0::2]
        valid[file] = acc[1::2]

        return np.array(train).astype(np.float32), np.array(valid).astype(np.float32)


f = plt.figure(figsize=(10, 8))
colors = sns.color_palette(n_colors=tasks)

train, valid = load_data(tasks=tasks)

print(valid)

plt.plot(valid, "--", label=r"Validation, prior: $U(a, b)$", color='maroon')
#plt.plot(i, valid, "--", label=r"Validation, prior: $q(w | \theta_A)$", color='navy')


plt.xlabel("Epochs")
plt.ylabel("Accuracy")
x_ticks = range(len(valid.shape[1]))
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))

f.suptitle("Evaluating continual learning")
plt.legend()

plt.savefig("results_continual.png")

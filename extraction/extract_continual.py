import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import seaborn as sns
import re
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=34)
plt.rcParams.update({'xtick.labelsize': 34, 'ytick.labelsize': 34, 'axes.labelsize': 34})

tasks = 2  # tasks continually learnt


def load_data(task):
    files = [open("diagnostics_{}.txt".format(task)).read() for task in range(task)]
    
    accuracies = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(\'acc\': \d.\d+)", file))) for file in files]
    valid = [acc[1::2] for acc in accuracies]

    return np.array(valid).astype(np.float32)


f = plt.figure(figsize=(15, 15))
colors = sns.color_palette(n_colors=tasks)

current_palette = sns.color_palette()
vt = load_data(task=tasks)
x_ticks = range(vt.shape[1])

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.xticks(x_ticks[9::10], map(lambda x: x + 1, x_ticks[9::10]))
f.suptitle("Evaluating continual learning")
plt.savefig("figures/results.png")

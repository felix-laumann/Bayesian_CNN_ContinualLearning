import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use("seaborn")
import seaborn as sns
import re
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=34)
plt.rcParams.update({'legend.fontsize': 34, 'xtick.labelsize': 34,
'ytick.labelsize': 34, 'axes.labelsize': 34})


def load_data(basename):
    files_no_transfer = [open("results/{0}_frac_{1}/diagnostics.txt".format(basename, frac)).read() for frac in ["0.10", "0.50","1.0"] ]
    
    accuracies = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(\'acc\': \d.\d+)", file))) for file in files_no_transfer]
    valid = [acc[1::2] for acc in accuracies]
    return np.array(valid).astype(np.float32)


f = plt.figure(figsize=(15, 15))
basename_to_plotname = {"task B": "No transfer learning. Fraction = ", "transfer": "With transfer learning. Fraction = "}
plot_type = {"task B": "-*", "transfer": "-^"}
colors = sns.color_palette(n_colors=4)
lgd = []
for basename in ["task B", "transfer"]:
    current_palette = sns.color_palette()
    vt = load_data(basename)
    x_ticks = range(vt.shape[1])
    legends = map(lambda x: "Fraction: " + x, ["0.10", "0.50", "1.0"])
    for values, legend_name, color in zip(vt, legends, colors):
        lgd.append(plt.plot(x_ticks, values, plot_type[basename], label=legend_name, color=color, markersize="20", markevery=15))

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

plt.xticks(x_ticks[49::50], map(lambda x: x + 1, x_ticks[49::50]))
#f.suptitle("Evalutating transfer properties")
first_legend= plt.legend(handles = list(map(lambda x: x[0], lgd[:3])), loc=10, title="Without transfer learning", ncol=2, bbox_to_anchor=(0.5, 0.7))
plt.gca().add_artist(first_legend)
plt.legend(handles = list(map(lambda x: x[0], lgd[3:])), ncol=2, loc=8, title="With transfer learning")
plt.savefig("figs/transfer_results.pdf")

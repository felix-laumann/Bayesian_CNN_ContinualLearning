import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import re
import numpy as np
plt.rc('font', family='serif', size=32)
plt.rcParams.update({'xtick.labelsize': 32, 'ytick.labelsize': 32, 'axes.labelsize': 32})

os.chdir("/home/felix/Dropbox/publications/Bayesian_CNN_continual/results/")

with open("diagnostics_1.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(acc)

train_1 = acc[1::2]
valid_1 = acc[0::2]
train_1 = np.array(train_1).astype(np.float32)
valid_1 = np.array(valid_1).astype(np.float32)

with open("diagnostics_2.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(acc)

train_2 = acc[1::2]
valid_2 = acc[0::2]
train_2 = np.array(train_2).astype(np.float32)
valid_2 = np.array(valid_2).astype(np.float32)

with open("diagnostics_2_eval.txt", 'r') as file:
    valid_2_eval_A = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(valid_2_eval_A)

valid_2_eval_A = np.array(valid_2_eval_A).astype(np.float32)

with open("diagnostics_3.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(acc)

train_3 = acc[1::2]
valid_3 = acc[0::2]
train_3 = np.array(train_3).astype(np.float32)
valid_3 = np.array(valid_3).astype(np.float32)

with open("diagnostics_3_eval.txt", 'r') as file:
    valid_3_eval_B = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(valid_3_eval_B)

valid_3_eval_B = np.array(valid_3_eval_B).astype(np.float32)
"""
with open("diagnostics_3_eval_A.txt", 'r') as file:
    valid_3_eval_A = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(valid_3_eval_A)

valid_3_eval_A = np.array(valid_3_eval_A).astype(np.float32)


with open("diagnostics_4.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(acc)

train_4 = acc[1::2]
valid_4 = acc[0::2]
train_4 = np.array(train_4).astype(np.float32)
valid_4 = np.array(valid_4).astype(np.float32)

with open("diagnostics_4_eval.txt", 'r') as file:
    valid_4_eval = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(valid_4_eval)

valid_4_eval = np.array(valid_4_eval).astype(np.float32)
"""
f = plt.figure(figsize=(20, 16))

plt.plot(valid_1, label=r"Validation task A , prior: $U(a, b)$", color='maroon')
plt.plot(valid_2, label=r"Validation task B, prior: $q(w | \theta_{A})$", color='darkblue')
plt.plot(valid_2_eval_A, label=r"Validation task A after training task B", color='#89c765')
plt.plot(valid_3, label=r"Validation task C, prior: $q(w | \theta_{B})$", color='peru')
plt.plot(valid_3_eval_B, label=r"Validation task B after training task C", color='m')
#plt.plot(valid_3_eval_A, label=r"Validation task A after training task C", color='gray')
#plt.plot(valid_4, "--", label=r"Validation, prior: $q(w | \theta_C)$", color='gray')
#plt.plot(valid_4_eval, "--", label=r"Validation task C after training task D", color='black')


plt.xlabel("Epochs")
plt.ylabel("Accuracy")
x_ticks = range(len(valid_1))
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))

plt.legend(loc='center right', fontsize=28)

# , bbox_to_anchor=(0.25, 0.58)

plt.savefig("results_continual.png", linewidth=10.0)

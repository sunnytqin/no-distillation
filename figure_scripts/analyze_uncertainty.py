'''
Visualize the entropy of the softmax output of the expert model over different epochs 
It generates visualizations for Figure 3 in the paper
To run: python analyze_uncertainty.py 
'''
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

epochs = [0, 11, 20, 50] # IPC 1
# epochs = ['90_expert1', '90_expert2', '90_expert3', '90_expert4', '90_expert5', '90_ensemble'] # IPC 50 ensemble
# epochs = [26, 46, 66] # IPC 10
ipc = 1
softlabels_all = []
for epoch in epochs:
    softlabels = torch.load(f"../softlabel/entropy/Tiny_random_ipc{ipc}_epoch{epoch}.pt")
    softlabels_all.append(softlabels.to("cpu"))

# load class info
# direct load class name if available
if os.path.exists("../softlabel/tiny_class_names.npy"):
    class_names_short = np.load("../softlabel/tiny_class_names.npy")

class_idx = 28 * ipc

# print top 3 guess at each epoch
for i in range(len(epochs)):
    print(f"Epoch {epochs[i]}")
    top3_idx = np.argsort(softlabels_all[i][class_idx].numpy())[-3:][::-1]
    print(class_names_short[top3_idx])


fig = plt.figure(figsize=(10, 2.3*len(epochs)))
plt.style.use('classic')
# set fontsize
plt.rcParams.update({'font.size': 20})
# three subplots 
for i in range(len(epochs)):
    plt.subplot(len(epochs), 1, i+1)
    plt.bar(range(200), softlabels_all[i].numpy()[class_idx,:], label=f'Expert at Epoch {epochs[i]}', alpha=0.5, color='C0',edgecolor='none')
    # show the class index to be red
    plt.bar(class_idx//ipc, softlabels_all[i].numpy()[class_idx, class_idx//ipc], color='r', edgecolor='none')
    plt.legend(loc='upper right', framealpha=0.5)
    plt.yticks(fontsize=15)
    if i == 0:
        plt.title(f'Soft labels for a {class_names_short[class_idx//ipc]} sample', fontsize=28, y=1.08)
    if i == len(epochs)-1:
        plt.xticks(np.arange(200), class_names_short, rotation=90, fontsize=3)
    else:
        plt.xticks([])
    # set y axis to show 3 ticks
    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
    plt.yticks(fontsize=18)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    plt.xlim([-0.5, 200.5])

# set a y label for a subplots
fig.text(0.01, 0.5, 'Softmax Probability', va='center', rotation='vertical', fontsize=25)
plt.xlabel('Class', fontsize=25)
plt.xticks(np.arange(200), class_names_short, rotation=90, fontsize=3)
plt.savefig(f"../softlabel/entropy/Tiny_ipc{ipc}_{class_names_short[class_idx//ipc].replace(' ', '_')}_class.pdf", dpi=300, bbox_inches='tight')
print(f"saved image: ../softlabel/entropy/Tiny_ipc{ipc}_{class_names_short[class_idx//ipc].replace(' ', '_')}_class.pdf")

'''
Some early analysis looking at entropy of labels (corret/incorrect, entropy change)
'''
import numpy as np
import matplotlib.pyplot as plt

''' per expert epoch analysis'''
# load the data 
dataset = 'Tiny'
model = 'ConvNetD4'
file_idx = 0
expert_epoch = 90

filename = f"entropy/label_analysis_{dataset}_{model}_expert{file_idx}_epoch{expert_epoch}.npz"

entropy_data = np.load(filename)
softlabels = entropy_data['softlabels']
hardlabels = entropy_data['hardlabels']
print("softlabels:", softlabels.shape)
# confirm probabilities sum to 1
assert np.all(np.abs(np.sum(softlabels, axis=1) - 1) < 1e-6)

fig, axs = plt.subplots(10, 1, figsize=(5, 20))
# set random seed for reproducibility
np.random.seed(0)
rand_idx = np.random.randint(0, softlabels.shape[1], 9)
for i, c in enumerate(rand_idx):
    idx = np.where(hardlabels==c)[0]
    # correct idx 
    predicted = np.argmax(softlabels[idx], axis=1)
    correct_idx = np.where(predicted==c)[0]
    
    correct_entropy = -np.sum(softlabels[idx][correct_idx] * np.log(softlabels[idx][correct_idx]), axis=1)
    entropy = -np.sum(softlabels[idx] * np.log(softlabels[idx]), axis=1)

    # plot historgram of softlabel
    axs[i].hist(correct_entropy, bins=50, histtype='step', label='Correct')
    axs[i].hist(entropy, bins=50, histtype='step', label='All')
    axs[i].legend()
    axs[i].set_title(f"Class {c}")
    axs[i].set_ylabel("Count")
    axs[i].set_xlim([0, 7])

# entropy for all classes
entropy = -np.sum(softlabels * np.log(softlabels), axis=1)
# plot historgram of softlabel
axs[9].hist(entropy, bins=50)
axs[9].set_title(f"All Data")
axs[9].set_xlabel("Entropy")
axs[9].set_ylabel("Count")
axs[9].set_xlim([0, 7])

# save the plot
plt.savefig(f"entropy/entropy_histogram_{dataset}_{model}_expert{file_idx}_epoch{expert_epoch}.png", 
            bbox_inches='tight', dpi=300)


''' summary statistics for each class '''
fig, axs = plt.subplots(1, 2, figsize=(20, 5))

mean_entropy = []
stdev_entropy = []

for i in range(softlabels.shape[1]):
    idx = np.where(hardlabels==i)[0]
    mean_entropy.append(np.mean(entropy[idx]))
    stdev_entropy.append(np.sqrt(np.var(entropy[idx])))

axs[0].scatter(np.arange(softlabels.shape[1]), mean_entropy)
axs[0].set_ylabel("Mean Entropy")
axs[0].set_xlabel("Class")

axs[1].scatter(np.arange(softlabels.shape[1]), stdev_entropy)
axs[1].set_ylabel("Stdev Entropy")
axs[1].set_xlabel("Class")

plt.title(f"Summary Statistics for Expert Epoch {expert_epoch}: {np.mean(entropy):.2f} +/- {np.sqrt(np.var(entropy)):.2f}")
plt.savefig(f"entropy/entropy_summary_{dataset}_{model}_expert{file_idx}_epoch{expert_epoch}.png", 
            bbox_inches='tight', dpi=300)


''' entropy progression over epochs '''
# # load the data 
# dataset = 'ImageNet64'
# model = 'ConvNetD4'
# file_idx = 0

# fig, axs = plt.subplots(6, 1, figsize=(5, 12))
# for i, epoch in enumerate([1, 2, 4, 6, 9, 15]):
#     filename = f"entropy/label_analysis_{dataset}_{model}_expert{file_idx}_epoch{epoch}.npz"
#     entropy_data = np.load(filename)
#     softlabels = entropy_data['softlabels']
#     hardlabels = entropy_data['hardlabels']

#     # entropy for all classes
#     entropy = -np.sum(softlabels * np.log(softlabels), axis=1)
#     # plot historgram of softlabel
#     axs[i].hist(entropy, bins=50)
#     axs[i].set_title(f"Epoch {epoch}")
#     axs[i].set_ylabel("Count")
#     axs[i].set_xlim([0, 7])
# axs[i].set_xlabel("Entropy")    

# # save the plot
# plt.savefig(f"entropy/entropy_histogram_{dataset}_{model}_expert{file_idx}_allepochs.png", 
#             bbox_inches='tight', dpi=300)
'''
Generate the indices for each class based on the cross entropy values
'''
import numpy as np
import torch
import matplotlib.pyplot as plt

# load the data 
dataset = 'Tiny'
model = 'ConvNetD4'
file_idx = 0
expert_epoch = 7

filename = f"entropy/label_analysis_{dataset}_{model}_expert{file_idx}_epoch{expert_epoch}.npz"

entropy_data = np.load(filename)
softlabels = entropy_data['softlabels']
hardlabels = entropy_data['hardlabels']

# convert softlabels and hardlabels to torch tensors
softlabels_tensor = torch.from_numpy(softlabels)
hardlabels_tensor = torch.from_numpy(hardlabels)

# compute cross entropy
cross_entropy = torch.nn.functional.cross_entropy(softlabels_tensor, hardlabels_tensor, reduction='none')

# convert cross entropy to numpy array
cross_entropy = cross_entropy.numpy()

predicted = np.argmax(softlabels, axis=1)
# entropy = -np.sum(softlabels * np.log(softlabels), axis=1)
print("softlabels:", softlabels.shape, cross_entropy.shape)

# confirm probabilities sum to 1
assert np.all(np.abs(np.sum(softlabels, axis=1) - 1) < 1e-6)

# set random seed for reproducibility
np.random.seed(0)
rand_idx = np.random.randint(0, softlabels.shape[1], 9)

# select images for each class
class_selection_dict = {}
cross_entropy_val_list = []
for c in range(softlabels.shape[1]):
# for c in range(2):
    class_idx = np.where(hardlabels==c)[0]
    # predicted_c = np.where(predicted==c)[0]
    # predicted_notc = np.where(predicted!=c)[0]
    # correct_idx = np.intersect1d(class_idx, predicted_c)
    # incorrect_idx = np.intersect1d(class_idx, predicted_notc)
    # if len(correct_idx) < 10:
    #     print(f"Class {c} has less than 10 correct samples")
    #     print(len(correct_idx), len(class_idx), len(incorrect_idx))
    #     continue

    # entropy to compute quantile value
    class_entropy = cross_entropy[class_idx]
    
    # correct_entropy = -np.sum(softlabels[idx][correct_idx] * np.log(softlabels[idx][correct_idx]), axis=1)

    # compute entropy quantiles for each class
    qs = np.linspace(0., 1., 11)
    q_val = []
    for q in qs:
        q_val.append(np.quantile(class_entropy, q))

    # index for each quantile
    class_quantile_idx_list = []
    class_quantile_val_list = []
    for s in range(10):
        elegible_idx = np.where((cross_entropy >= q_val[s]) & (cross_entropy < q_val[s+1]))[0]
        class_quantile_idx = np.intersect1d(elegible_idx, class_idx)
        class_quantile_idx_list.append(class_quantile_idx)
    
        # compute average cross entropy for each quantile
        class_quantile_val_list.append(np.mean(cross_entropy[class_quantile_idx]))
        assert q_val[s] < np.mean(cross_entropy[class_quantile_idx])
        assert q_val[s+1] >= np.mean(cross_entropy[class_quantile_idx])
    cross_entropy_val_list.append(class_quantile_val_list)

    
    # # stratefied samples
    # stratefied_idx = []
    # for s in range(10):
    #     elegible_idx = class_quantile_idx_list[s][0] # select one from each qunatile
    #     stratefied_idx.append(elegible_idx)

    # save the indices
    class_selection_dict[c] = {}
    for s in range(10):
        class_selection_dict[c][str(s)] = class_quantile_idx_list[s]
    # class_selection_dict[c]['stratefied'] = stratefied_idx

# compute average cross entropy for each quantile
cross_entropy_val_list = np.array(cross_entropy_val_list)
print(cross_entropy_val_list.shape)
cross_entropy_avg = np.mean(cross_entropy_val_list, axis=0)
print(cross_entropy_avg)
    
# save dictionary
np.save(f"entropy/label_analysis_{dataset}_{model}_expert{file_idx}_epoch{expert_epoch}_class_selection_dict.npy", class_selection_dict)



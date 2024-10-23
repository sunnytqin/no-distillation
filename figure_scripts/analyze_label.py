'''
Visualize clusters in softmax density,
and get class index for data-efficiency test
'''
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import os

''' per expert epoch analysis'''
# load the data 
dataset = 'Tiny'
# load_checkpoint = 'Tiny_mtt_ipc10.pt'
load_checkpoint = None
data_path = '/n/home05/sqin/invariance/data/tiny-imagenet-200'
num_classes = 200
model = 'ConvNetD4'
file_idx = 0
expert_epoch = 50

if load_checkpoint is None:
    filename = f"entropy/label_analysis_{dataset}_{model}_expert{file_idx}_epoch{expert_epoch}.npz"
else:
    filename = f"entropy/label_analysis_{dataset}_{load_checkpoint}_{model}_expert{file_idx}_epoch{expert_epoch}.npz"

# load entropy data
entropy_data = np.load(filename)
softlabels = entropy_data['softlabels']
hardlabels = entropy_data['hardlabels']
print("softlabels:", softlabels.shape)
# confirm probabilities sum to 1
assert np.all(np.abs(np.sum(softlabels, axis=1) - 1) < 1e-6)

# load class info
# direct load class name if available
if os.path.exists("tiny_class_names.npy"):
    class_names_short = np.load("tiny_class_names.npy")
    print("directly loaded class names")
else:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
    # dst_test = datasets.ImageFolder(os.path.join(data_path, "val", "images"), transform=transform)
    class_ids = dst_train.classes

    # load class name txt
    name_dict = {}
    with open(os.path.join(data_path, "words.txt"), 'r') as f:
        class_name_txt = f.readlines()
        for f in class_name_txt:
            idx, name = f.replace('\n', '').split('\t')
            name_dict[idx] = name

    class_names = []
    class_names_short = []
    for id in class_ids:
        class_names.append(name_dict[id])
        class_names_short.append(name_dict[id].split(',')[0])

    # save class names for furture use
    # print(class_names_short[0:10])
    # np.save("tiny_class_names.npy", class_names_short)


# for each predicted class, look at the second guess
first_guess = np.argmax(softlabels, axis=1)
# second_guess = np.argsort(softlabels)[:, :-1]
# print("second guess indices:", second_guess.shape)

density_matrix = []
for i in range(num_classes):
    cls_idx = np.where(hardlabels==i)[0] # based on true class
    # cls_idx = np.where(first_guess==i)[0] # based on expert guess
    cls_probs = np.mean(softlabels[cls_idx], axis=0) 
    cls_probs[i] = np.nan 
    density_matrix.append(cls_probs)

density_matrix = np.array(density_matrix)
# generate classes for data efficiency study
train_acc = np.zeros(num_classes)
for i in range(num_classes):
    cls_idx = np.where(hardlabels==i)[0]
    train_acc[i] = np.sum(first_guess[cls_idx] == i) / len(cls_idx)
    # print(i, class_names_short[i], train_acc[i])

# generate class selection for data efficiency study
# idx_sort = np.argsort(train_acc)[::-1]
# control_list = []
# label_name = []
# for i in range(0, 200, 5):
#     # print the most confused class
#     idx = idx_sort[i]
#     control_list.append(idx)
#     label_name.append(f"{class_names_short[idx]}")

# np.savez(f"entropy/{dataset}_{model}_epoch{expert_epoch}_data_efficient.npz",
#          subject=control_list, label_name=label_name)

# compute the entropy of softlabels
entropy = np.mean(-np.sum(softlabels * np.log(softlabels + 1e-6), axis=1))

# heat map of density_matrix
plt.figure(figsize=(10, 10))
plt.imshow(density_matrix, cmap="magma_r", interpolation='none')
plt.colorbar()
plt.title(f"Training data IPC=10\nNon-top softmax probability (Entropy={entropy:.1f})")
plt.xlabel("Non-top guess")
plt.ylabel("First Guess Class")
plt.xticks(np.arange(num_classes), class_names_short, rotation=90, fontsize=3)
plt.yticks(np.arange(num_classes), class_names_short, fontsize=3)
if load_checkpoint is None:
    plt.savefig(f"entropy/second_guess_density_{dataset}_{model}_expert{file_idx}_epoch{expert_epoch}.png", 
            bbox_inches='tight', dpi=300)
    print("saved visualization: ", f"entropy/second_guess_density_{dataset}_{model}_expert{file_idx}_epoch{expert_epoch}.png")

else:
    plt.savefig(f"entropy/second_guess_density_{dataset}_{load_checkpoint}_{model}_expert{file_idx}_epoch{expert_epoch}.png", 
            bbox_inches='tight', dpi=300)
    print("saved visualization: ", f"entropy/second_guess_density_{dataset}_{load_checkpoint}_{model}_expert{file_idx}_epoch{expert_epoch}.png")

import os
import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torchvision import datasets, transforms, models


# load name
class_names_short = np.load("../tag_expert/tiny_class_names.npy")
# for i in range(len(class_names_short)):
#     print(i, class_names_short[i])

# 28 German shepherd
# 32 Egyptian cat
# 66 barn
# 96 dam
# 161 teddy
# 188 banana
# 195 cliff

def create_image_grid(img_tenosr, save_name):
    # create torch image grid
    class_idx = [28, 32, 66, 96, 161, 188, 195]
    img_tenosr = torch.nn.functional.interpolate(img_tenosr[class_idx], size=256, mode='bilinear', align_corners=False)
    grid = torchvision.utils.make_grid(img_tenosr, nrow=len(class_idx), normalize=True, scale_each=False, padding=15, pad_value=1)
    # save the grid
    torchvision.utils.save_image(grid, save_name)
    return

def create_individual_grid():
    mean = np.array([0.485, 0.456, 0.406])
    std= np.array([0.229, 0.224, 0.225])
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [1./i for i in std]),
                                transforms.Normalize(mean = [-i for i in mean],
                                                        std = [ 1., 1., 1. ]),
                                                        ])

    ''' BPTT imgages '''
    checkpoint_name = '../../Simple_Dataset_Distillation/save/tiny/IPC_1_0_curr_unroll_90_130_0_softlabel2.h5'

    with h5py.File(checkpoint_name, 'r') as file:
        # Access the dataset
        bptt_images = file['data']
        epoch = file['epoch']

        # reshape the data 
        bptt_images = torch.tensor(np.array(bptt_images).reshape(-1, 3, 64, 64))

    print("bptt_imgs:", bptt_images.shape, type(bptt_images))
    create_image_grid(bptt_images, "bptt.png")

    ''' training data imgages '''
    training_imgs = torch.load(os.path.join(f"/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data/Tiny/ipc1_random_image_syn.pt")).cpu()
    print("training_imgs:", training_imgs.shape)


    training_imgs = invTrans(training_imgs)
    create_image_grid(training_imgs, "training.png")
    ''' SRe2L imgages '''

    # SRe2L images
    data_path = '/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data/sre2l_tiny_rn18_1k_ipc50'
    transform = transforms.Compose([transforms.ToTensor()])
    sre2l_data = datasets.ImageFolder(data_path, transform=transform) # no augmentation

    sre2l_imgs = []
    for i in range(0, len(sre2l_data), 50):
        sre2l_imgs.append(sre2l_data[i][0])
    sre2l_imgs = torch.tensor(np.array(sre2l_imgs))
    print("sre2l_imgs:", sre2l_imgs.shape)
    create_image_grid(sre2l_imgs, "sre2l.png")

    # MTT images
    data_path = '/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data/Tiny_mtt_ipc10.pt'
    mtt_imgs = torch.load(data_path)
    mtt_imgs = invTrans(mtt_imgs)
    # select every 10 images
    mtt_imgs = mtt_imgs[::10]
    print("mtt_imgs:", mtt_imgs.shape)
    create_image_grid(mtt_imgs, "mtt.png")


# load all the pngs 
methods = ['training', 'bptt', 'sre2l', 'mtt']
method_names = ['Data', 'BPTT', r'SRe$^2$L', 'MTT']
method_imgs = []
for m in methods:
    img = plt.imread(f"{m}.png")
    method_imgs.append(img)

# use matplotlib to combine them into one plot
fig, ax = plt.subplots(4, 1, figsize=(40, 10))
for i in range(4):
    ax[i].axis('off')
    ax[i].imshow(method_imgs[i])
plt.subplots_adjust(hspace=0.05)
plt.savefig("image_grid.pdf", bbox_inches='tight', dpi=500)


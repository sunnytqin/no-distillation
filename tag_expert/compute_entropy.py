import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import augment, get_dataset, get_network, get_eval_pool, ParamDiffAug, get_time
import wandb
import copy
import random
from reparam_module import ReparamModule

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        raise ValueError("GPU required")
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    if args.dsa:
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None
    
    # if args.wandb:
    #     mode = "online"
    # else:
    #     mode = "disabled"

    wandb.init(sync_tensorboard=False,
               project="data-softlabel",
               entity="dotml", 
               config=args,
               name=args.run_name,
               notes=args.notes,
               mode="disabled",
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans
    
    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    indices_class = [[] for c in range(num_classes)]
    # Build label to index map
    print("---------------Build label to index map--------------")
    # For machines with limited RAM, it's impossible to load all ImageNet or even TinyImageNet into memory.
    # Even if it's possible, it will take too long to process.
    # Therefore we pregenerate an indices to image map and use this map to quickly random samples from ImageNet or TinyImageNet dataset.
    if args.dataset == 'ImageNet':
        indices_class = np.load('indices/imagenet_indices_class.npy', allow_pickle=True)
    elif args.dataset == 'Tiny':
        indices_class = np.load('indices/tiny_indices_class.npy', allow_pickle=True)
    else:
        for i, data in tqdm(enumerate(dst_train)):
            indices_class[data[1]].append(i)
        
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=False, num_workers=16)

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    elif args.dataset == "ImageNet" and (not args.divide_epoch):
        expert_dir = os.path.join(expert_dir, args.model)
    elif args.dataset == "ImageNet" and args.divide_epoch:
        expert_dir = os.path.join(args.buffer_path, "ImageNet_DIVIDE_EPOCH")
        expert_dir = os.path.join(expert_dir, 'ConvNetD4')
    elif args.dataset == "ImageNet64" and args.model=="ConvNetD4":
        expert_dir = os.path.join(expert_dir, 'imagenette', '64', 'ConvNet')
    elif args.dataset == "ImageNet64" and args.model=="ResNet18_AP":
        expert_dir = os.path.join(expert_dir, 'imagenette', '64', args.model)
    else: 
        raise AssertionError("Unknown dataset")
    print("Expert Dir: {}".format(expert_dir))

    if not args.random_trajectory:
        if args.load_all:
            buffer = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))

        else:
            expert_files = []
            n = 1
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            file_idx = 0
            expert_idx = 0
            if not args.fixed_expert:
                random.shuffle(expert_files)
            if args.max_files is not None:
                expert_files = expert_files[:args.max_files]
            print("loading file {}".format(expert_files[file_idx]))
            buffer = torch.load(expert_files[file_idx])

    expert_trajectory = buffer[expert_idx]

    start_epoch = args.max_start_epoch

    target_params = expert_trajectory[start_epoch]

    target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

    # produce soft labels for each image.
    label_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
    label_net = ReparamModule(label_net)
    label_net.eval()

    # use the target param as the model param to get soft labels.
    label_params = copy.deepcopy(target_params.detach()).requires_grad_(False)

    softlabels = []
    hardlabels = []
    for i_batch, datum in enumerate(trainloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].to(args.device)

        output = label_net(img, flat_param=label_params)
        label_syn = torch.nn.functional.softmax(output, dim=1)
        softlabels.append(label_syn)
        hardlabels.append(lab)
    # save the soft labels as numpy
    softlabels = torch.cat(softlabels, 0).detach().cpu().numpy()
    hardlabels = torch.cat(hardlabels, 0).detach().cpu().numpy()

    np.savez(f"entropy/label_analysis_{args.dataset}_{args.model}_expert{file_idx}_epoch{args.max_start_epoch}", softlabels=softlabels, hardlabels=hardlabels)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=3, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=10000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='/nfs/data/justincui/data/tiny-imagenet-200', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--random_trajectory', action='store_true', default=False, help="using random trajectory instead of pretrained")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--teacher_label', action='store_true', default=False, help='whether to use label from the expert model to guide the distillation process.')
    parser.add_argument('--run_name', type=str, default=None, help='wandb expt name')
    parser.add_argument('--notes', type=str, default="No description", help='wandb additional notes')
    parser.add_argument('--load_checkpoint', type=str, default=None, help="load pretrained")

    parser.add_argument('--temp', type=float, default=1.0, help='teacher label temp')
    parser.add_argument('--wandb', action='store_true', help="logging")
    parser.add_argument('--fixed_expert', action='store_true', help="will disable expert shuffling")
    parser.add_argument('--divide_epoch', action='store_true', help="finer grained expert epoch trajectories")
    

    args = parser.parse_args()

    main(args)



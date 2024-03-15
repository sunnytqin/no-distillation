import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import augment, get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, DiffAugmentList, ParamDiffAug, get_memory
import wandb
import copy
import random
from reparam_module import ReparamModule
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        raise ValueError("GPU required")
    eval_it_pool = np.arange(1, args.Iteration + 1, args.eval_it).tolist()
    # eval_it_pool = np.arange(1000, args.Iteration + 1, args.eval_it).tolist() # no need to do evaluation in the first 500 iterations
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None
    
    if args.wandb:
        mode = "online"
    else:
        mode = "disabled"

    wandb.init(sync_tensorboard=False,
               project="data-softlabel",
               entity="dotml", 
               config=args,
               name=args.run_name,
               notes=args.notes,
               mode=mode,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1
    # args.distributed = False


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

    # for c in range(num_classes):
    #     print('class c = %d: %d real images'%(c, len(indices_class[c])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        subset = Subset(dst_train, idx_shuffle)
        data_loader = DataLoader(subset, batch_size=n)
        # only read the first batch which has n(IPC) number of images.
        for data in data_loader:
            return data[0].to("cpu")


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    # get_memory("before syndata assignment")

    # save image_syn for future use
    # torch.save(image_syn, os.path.join("/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data", "ipc10_random_image_syn.pt"))
    
    if args.load_checkpoint is not None:
        print("loading synthetic images from {}".format(args.load_checkpoint))
        image_syn = torch.load(os.path.join("/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data", args.load_checkpoint))
    else: 
        # if args.pix_init == 'real':
        # print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
        # else:
        #     print('initialize synthetic data from random noise')

    ''' training '''
    # image_syn = image_syn.detach().to(args.device).requires_grad_(False)
    # syn_lr = syn_lr.detach().to(args.device).requires_grad_(False)
    # # check which devicde image_syn is on
    # print("image_syn device: ", image_syn.device)

    # criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

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
            if args.max_experts is not None:
                buffer = buffer[:args.max_experts]
            if not args.fixed_expert:
                random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    ''' generate experiment hyperparameters '''

    ''' start epoch only'''
    tune_start_epoch = np.arange(args.max_start_epoch, args.max_start_epoch+1, 1).tolist()
    expt_dict = {}
    for i, se in enumerate(tune_start_epoch):
        expt_dict[i] = se
    print("Total number of experiments: {}".format(len(expt_dict)))
    print("start epoch list: {}".format(expt_dict))

   
    ''' temperature and start epoch grid '''
    # tune_start_epoch = np.arange(6, args.max_start_epoch + 2, 2).tolist()
    # tune_temp = [1.0, 0.9, 0.8, 0.6, 0.4, 0.2]
    # expt_dict = {}
    # expt_num = 0
    # for se in tune_start_epoch:
    #     for t in tune_temp:
    #         expt_dict[expt_num] = (se, t)
    #         expt_num += 1
    # print("Total number of experiments: {}".format(len(expt_dict)))
    # print("grid: {}".format(expt_dict))
    
    ''' learning rate '''
    # expt_dict = {}
    # lr_list = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    # for i, lr in enumerate(lr_list):
    #     expt_dict[i] = lr
    # print("Total number of experiments: {}".format(len(expt_dict)))
    # print("lr list: {}".format(expt_dict))

    for exp_num in range(0, args.Iteration+1):
        # save_this_it = False
        args.exp_num = exp_num
        
        ''' assign hparams'''
        ''' start epoch only'''
        args.max_start_epoch = expt_dict[exp_num]

        ''' start epoch and temp grid'''
        # args.max_start_epoch, args.temp = expt_dict[exp_num]

        ''' lr '''
        # args.lr_teacher = expt_dict[exp_num]

        print('-------------------------\nExperiment ID = %d'%exp_num)
        # writer.add_scalar('Progress', it, it)
        
        wandb.log({'Experiment': exp_num,
                   'Start_Epoch': args.max_start_epoch,
                   'Temp': args.temp,
                   'Expert ID': file_idx})
        
        # softlabel assignment
        # get_memory("before softlabel assignment")
        if not args.fixed_expert or exp_num == 0: 
            if not args.random_trajectory:
                if args.load_all:
                    expert_trajectory = buffer[np.random.randint(0, len(buffer))]
                else:
                    expert_trajectory = buffer[expert_idx]
                    expert_idx += 1
                    if expert_idx == len(buffer):
                        expert_idx = 0
                        file_idx += 1
                        if file_idx == len(expert_files):
                            file_idx = 0
                            if not args.fixed_expert:
                                random.shuffle(expert_files)
                        # print("loading file {}".format(expert_files[file_idx]))
                        # if args.max_files != 1:
                        #     del buffer
                        #     buffer = torch.load(expert_files[file_idx])
                        # if args.max_experts is not None:
                        #     buffer = buffer[:args.max_experts]
                        if not args.fixed_expert:
                            random.shuffle(buffer)

        start_epoch = args.max_start_epoch

        # if not args.random_trajectory:
        #     starting_params = expert_trajectory[start_epoch]
 
        #     target_params = expert_trajectory[start_epoch]
        # else:
        #     starting_params = [p for p in student_net.parameters()]
        #     target_params = [p for p in student_net.parameters()]
        target_params = expert_trajectory[start_epoch]

        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        # produce soft labels for soft label assignment.
        if args.teacher_label:
            label_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
            label_net = ReparamModule(label_net)
            label_net.eval()
            # get_memory("got teacher model")

            # use the target param as the model param to get soft labels.
            label_params = copy.deepcopy(target_params.detach()).requires_grad_(False)

            batch_labels = []
            SOFT_INIT_BATCH_SIZE = 50
            if image_syn.shape[0] > SOFT_INIT_BATCH_SIZE and args.dataset in ['ImageNet64', 'ImageNet']:
                for indices in torch.split(torch.tensor([i for i in range(0, image_syn.shape[0])], dtype=torch.long), SOFT_INIT_BATCH_SIZE):
                    batch_labels.append(label_net(image_syn[indices].detach().to(args.device), flat_param=label_params))
            else:
                label_syn = label_net(image_syn.detach().to(args.device), flat_param=label_params)
            label_syn = torch.cat(batch_labels, dim=0)
            label_syn = torch.nn.functional.softmax(label_syn * args.temp)
            del label_net, label_params
            for _ in batch_labels:
                del _
            torch.cuda.empty_cache()
        # get_memory("softlabel assignment done")

        ''' Evaluate synthetic data '''
        # if it in eval_it_pool and args.eval_it > 0:
        for model_eval in model_eval_pool:
            print('-------------------------\nEvaluation\nExperiment ID = %d, model_train = %s, model_eval = %s'%(args.exp_num, args.model, model_eval))
            if args.dsa:
                print('DSA augmentation strategy: \n', args.dsa_strategy)
                print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
            else:
                print('DC augmentation parameters: \n', args.dc_aug_param)

            accs_test = []
            accs_train = []
            for it_eval in range(args.num_eval):
                # get_memory("before student network definition")
                net_eval = get_network(model_eval, channel, num_classes, im_size, dist=True)  #.to(args.device) # get a random model
                # get_memory("after student network definition")
                
                # # count number of parameters in net_eval
                # num_params = sum(p.numel() for p in net_eval.parameters() if p.requires_grad)
                # print("ConvNetD4 params:", num_params)

                # eval_labs = label_syn
                # with torch.no_grad():
                #     image_save = image_syn
                # image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                args.lr_net = syn_lr.item()
                # _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn, label_syn, testloader, args, texture=args.texture)
                accs_test.append(acc_test)
                accs_train.append(acc_train)
            accs_test = np.array(accs_test)
            accs_train = np.array(accs_train)
            acc_test_mean = np.mean(accs_test)
            acc_test_std = np.std(accs_test)
            if acc_test_mean > best_acc[model_eval]:
                best_acc[model_eval] = acc_test_mean
                best_std[model_eval] = acc_test_std
                # save_this_it = True
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
            # delete network, clear cache
            del net_eval
            torch.cuda.empty_cache()

            wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean})
            wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]})
            wandb.log({'Std/{}'.format(model_eval): acc_test_std})
            wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]})

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
    parser.add_argument('--optimizer', type=str, default='SGD', choices=["SGD", "AdamW"], help='optmizer to train a model with distilled data')
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



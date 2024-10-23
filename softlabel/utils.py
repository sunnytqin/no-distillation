# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import kornia as K
import tqdm

sys.path.append('../train_expert/')
import presets
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torchvision.transforms.functional import InterpolationMode
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import ConvNet
import pickle
import copy
import wandb


def get_dataset(dataset, data_path, batch_size=1, args=None):

    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val", "images"), transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'ImageNet':
        channel = 3
        im_size = (224, 224)
        num_classes = 1000

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # pytorch official implementation for ResNet
        data_transforms = {
            "train": presets.ClassificationPresetTrain(
                crop_size=224,
                interpolation=InterpolationMode("bilinear"),
                auto_augment_policy=None,
                random_erase_prob=0.0,
                ra_magnitude=9,
                augmix_severity=3,
                backend="PIL",
                use_v2=False,
            ),

            "val": presets.ClassificationPresetEval(
            crop_size=224,
            resize_size=256,
            interpolation=InterpolationMode("bilinear"),
            backend="PIL",
            use_v2=False,
            )
        }
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=data_transforms['train']) # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=data_transforms['val'])
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}
        class_map_inv = class_map

    elif dataset == 'ImageNet64':
        channel = 3
        im_size = (64, 64)
        args.res = 64
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x_train = []
        y_train = []
        for idx in range(1, 11):

            data_file = os.path.join(data_path, 'train_data_batch_')
            
            with open(data_file + str(idx), 'rb') as fo:
                d = pickle.load(fo)

            x_train.append(d['data'].reshape(-1, 3, 64, 64))
            y_train.append(d['labels'])

        x_train = np.concatenate(x_train) 
        y_train = np.concatenate(y_train) 

        x_train = torch.from_numpy(x_train).to(torch.float32) / 255. 
        y_train = torch.from_numpy(y_train) - 1

        # validation set 
        data_file = os.path.join(data_path, 'val_data')

        with open(data_file, 'rb') as fo:
                d = pickle.load(fo)
        x_test = d['data'].reshape(-1, 3, 64, 64)
        y_test = d['labels']
        x_test = torch.from_numpy(x_test).to(torch.float32) / 255. 
        y_test = torch.as_tensor(y_test) - 1

        mean = [0.485, 0.456, 0.406]
        std = np.array([0.229, 0.224, 0.225]) 

        data_transforms = transforms.Normalize(mean=mean, std=std)
        x_train = data_transforms(x_train)
        dst_train = TensorDataset(copy.deepcopy(x_train.detach()),copy.deepcopy( y_train.detach()))

        data_transforms = transforms.Normalize(mean=mean, std=std)
        x_test = data_transforms(x_test)
        dst_test = TensorDataset(x_test.detach(), y_test.detach())
        class_names = None
        class_map = {x: x for x in range(num_classes)}
    
    elif dataset == 'SRe2L':
        channel = 3
        im_size = (224, 224)
        num_classes = 1000

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # pytorch official implementation for ResNet
        data_transforms = {
            "train": presets.ClassificationPresetTrain(
                crop_size=224,
                interpolation=InterpolationMode("bilinear"),
                auto_augment_policy=None,
                random_erase_prob=0.0,
                ra_magnitude=9,
                augmix_severity=3,
                backend="PIL",
                use_v2=False,
            ),

            "val": presets.ClassificationPresetEval(
            crop_size=224,
            resize_size=256,
            interpolation=InterpolationMode("bilinear"),
            backend="PIL",
            use_v2=False,
            )
        }
        dst_train = datasets.ImageFolder("/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data/sre2l_in1k_rn18_4k_ipc200", transform=data_transforms['train']) # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=data_transforms['val'])
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}
        class_map_inv = class_map

    elif dataset == 'BPTT':
        channel = 3
        im_size = (64, 64)
        num_classes = 1000

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        data_transforms = {
            'train': transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.Resize(im_size, antialias=None),
                                            transforms.CenterCrop(im_size)]),
            'val': transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.Resize(im_size, antialias=None),
                                            transforms.CenterCrop(im_size)]),
        }

        # dst_train = datasets.ImageFolder("/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data/sre2l_in1k_rn18_4k_ipc200", transform=data_transforms['train']) # no augmentation
        dst_train = None
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=data_transforms['val'])
        class_names = dst_test.classes
        class_map = {x:x for x in range(num_classes)}
        class_map_inv = class_map

    elif dataset.startswith('CIFAR100'):
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), transforms.Resize(im_size)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}

    else:
        exit('unknown dataset: %s'%dataset)

    if args.zca:
        images = []
        labels = []
        print("Train ZCA")
        for i in tqdm.tqdm(range(len(dst_train))):
            im, lab = dst_train[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        zca.fit(images)
        zca_images = zca(images).to("cpu")
        dst_train = TensorDataset(zca_images, labels)

        images = []
        labels = []
        print("Test ZCA")
        for i in tqdm.tqdm(range(len(dst_test))):
            im, lab = dst_test[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")

        zca_images = zca(images).to("cpu")
        dst_test = TensorDataset(zca_images, labels)

        args.zca_trans = zca


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1)


    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv


def get_imagenet_class_name():
    class_dict = {}
    with open("imagenet1000_clsidx_to_labels.txt") as f:
        for line in f:
            (key, val) = line.split(":")
            key = key.replace('{', '').strip()
            val = val.replace('}', '').replace(',', '').replace("'", '').strip()
            class_dict[int(key)] = val
    return class_dict

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size=(32, 32), dist=True, **kwargs):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ResNet18':
        net = models.__dict__['resnet18']()
    elif model == 'ResNet50':
        net = models.__dict__['resnet50']()
    # elif model == 'ConvNet_MoE':
    #     net = ConvNet_MoE(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    else:
        raise ValueError('Error: unknown model.')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def get_memory(message): 
    # check GPU memory usage
    print(get_time(), message)
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    return


def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img =  datum[0].float().to(args.device)
        lab = datum[1].to(args.device)

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)

        if (mode == 'train' and args.teacher_label) or (mode == 'train' and len(datum[1].shape)> 1) :
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), np.argmax(datum[1].cpu().data.numpy(), axis=-1)))
        else:
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
            if args.selection_strategy in ['data_efficient_treat_label','data_efficient_treat_image', 'data_efficient_control']:
                # only examine treatment subject
                subject_idx = torch.where(lab == args.treat_subject)[0].cpu().data.numpy()
                n_b = len(subject_idx)
                acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1)[subject_idx], lab.cpu().data.numpy()[subject_idx]))
               
        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss=False):
    net = net.to(args.device)
    if args.ipc <= 100: # load to device for faster inference
        images_train = images_train.to(args.device)
        labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        scheduler = StepLR(optimizer, step_size=args.epoch_eval_train//4, gamma=0.3)

    elif args.optimizer == 'AdamW':
        # adam optimizer - following SRe2L hparams, mostly used for ResNets
        optimizer = torch.optim.AdamW(net.parameters(),
                                        lr=lr,
                                        weight_decay=0.01)
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epoch_eval_train)) if step <= args.epoch_eval_train else 0, last_epoch=-1)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []
    
    max_acc_test = 0.0
    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch or (ep % 200 == 0 and ep > 0):
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
                wandb.log({f'Accuracy/Exp_{args.exp_num}/Test': acc_test})
                if acc_test > max_acc_test:
                    max_acc_test = acc_test
                # print('%s Evaluate_%02d: epoch = %04d  train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, loss_train, acc_train, acc_test))
        scheduler.step()
        wandb.log({f'Accuracy/Exp_{args.exp_num}/Train': acc_train})
        wandb.log({f'Accuracy/Exp_{args.exp_num}/Step': ep})

    time_train = time.time() - start
    if args.selection_strategy in ['data_efficient_treat_image','data_efficient_treat_label', 'data_efficient_control']:
        max_acc_test = acc_test

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, max_acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, max_acc_test


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x

def DiffAugmentList(x_list, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    for x in x_list:
                        x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                for x in x_list:
                    x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        for x in x_list:
            x = x.contiguous()
    return x_list


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}
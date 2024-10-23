import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_dataset, get_network, evaluate_synset, get_time, ParamDiffAug, get_memory
import wandb
import copy
from reparam_module import ReparamModule
from torch.utils.data import DataLoader, Subset

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    # set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        raise ValueError("GPU required")

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)

    args.im_size = im_size 

    if args.dsa:
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
               entity="", 
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

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model: ', args.student_model)

    ''' organize the real dataset '''
    indices_class = [[] for c in range(num_classes)]
    # Build label to index map
    print("---------------Build label to index map--------------")
    # for random image selection
    if args.dataset == 'ImageNet':
        indices_class = np.load('indices/imagenet_indices_class.npy', allow_pickle=True)
    elif args.dataset == 'Tiny':
        indices_class = np.load('indices/tiny_indices_class.npy', allow_pickle=True)
    elif args.dataset == "SRe2L":
        indices_class = np.load('indices/imagenet_sre2l_class.npy', allow_pickle=True).item()
    elif args.dataset == 'BPTT':
        indices_class = None # not used 
    else:
        for i, data in tqdm(enumerate(dst_train)):
            indices_class[data[1]].append(i)
        # save for future use 
        # indices_class_dict = {}
        # for c in range(num_classes):
        #     indices_class_dict[c] = indices_class[c]
        # np.save(f'indices/{args.dataset}_indices_class.npy', indices_class_dict)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        subset = Subset(dst_train, idx_shuffle)
        data_loader = DataLoader(subset, batch_size=n) # only read the first batch which has n(IPC) number of images.
        for data in data_loader:
            return data[0].to("cpu")
        
    def select_images(class_selection_dict, c, n, difficulty): # Image selection w/ Cross-Entropy
        idx_selected = class_selection_dict[c][difficulty]
        if len(idx_selected) < n:
            if int(difficulty) < 9:
                idx_selected_extend = class_selection_dict[c][str(int(difficulty)+1)][:n-len(idx_selected)]
            else:
                # select a random image
                idx_selected_extend = np.random.choice(indices_class[c], n-len(idx_selected), replace=False)
            idx_selected = np.concatenate([idx_selected, idx_selected_extend])
        assert len(idx_selected) >= n
        subset = Subset(dst_train, idx_selected)
        data_loader = DataLoader(subset, batch_size=n)
        # only read the first batch which has n(IPC) number of images.
        for data in data_loader:
            return data[0].to("cpu")

    ''' initialize the synthetic data '''
    if args.load_checkpoint is not None:
        print("loading synthetic images from {}".format(args.load_checkpoint))
        if ".pt" in args.load_checkpoint:
            image_syn = torch.load(os.path.join(f"/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data/{args.dataset}", args.load_checkpoint))
        elif ".h5" in args.load_checkpoint: # BPTT learned images 
            import h5py
            # with h5py.File(os.path.join('/n/home05/sqin/softlabels/Simple_Dataset_Distillation/save/tiny', args.load_checkpoint), 'r') as file:
            with h5py.File(os.path.join('/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data', args.load_checkpoint), 'r') as file:  
                image_syn = torch.tensor(np.array(file['data']).reshape(-1, 3, 64, 64))
                # label_syn = torch.argmax(torch.tensor(np.array(file['label'])), axis=1) # hardlabel
                label_syn = torch.tensor(np.array(file['label'])) # softlabel

    else: 
        if args.selection_strategy == 'random':
            print('initialize synthetic data from random real images')
            image_syn = torch.zeros(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
            # save image_syn for future use
            # torch.save(image_syn, os.path.join(f"/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data/{args.dataset}", f"ipc{args.ipc}_random_image_syn.pt"))

        elif args.selection_strategy in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 'stratefied']:
            class_selection_dict = np.load(f"entropy/label_analysis_{args.dataset}_{args.teacher_model}_expert0_epoch{args.max_expert_epoch}_class_selection_dict.npy", allow_pickle=True).item()
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = select_images(class_selection_dict, c, args.ipc, args.selection_strategy).detach().data
        
        elif args.selection_strategy in ['data_efficient_treat_label','data_efficient_treat_image', 'data_efficient_control', 'data_knowledge_scaling']:
            pass # images and labels will be assigned later

        else:
            raise ValueError("Unknown selection strategy")
        
    def reassign_image_syn(): 
        # read swap file - sawp file has a list of class index that covers different class difficulties (from super easy class to super hard class)
        # you can also random select a class to conduct experiment, but I just want to make sure I cover the entire range of classes based on test accuracy
        if args.ipc == 1:
            swap_info = np.load('entropy/Tiny_ConvNetD4_epoch7_data_efficient.npz')
        elif args.ipc == 10:
            swap_info = np.load('entropy/Tiny_ConvNetD4_epoch50_data_efficient.npz')
        elif args.ipc == 50:
            swap_info = np.load('entropy/Tiny_ConvNetD4_epoch90_data_efficient.npz')
        else: 
            raise ValueError("Not implemented")
        # load pre-saved random images
        image_syn = torch.load(os.path.join(f"/n/holyscratch01/barak_lab/Lab/sqin/invariance/softlabel_syn_data/{args.dataset}", f"ipc{args.ipc}_random_image_syn.pt"))
        
        # select treatment subject
        treat_subject = swap_info['subject'][args.selection_idx]

        args.treat_subject = treat_subject
        print("Treat subject: ", args.treat_subject, "Strategy: ", args.selection_strategy)
        if args.selection_strategy == 'data_efficient_treat_image': 
            # remove image treatment: remove subject data
            mask = torch.ones(len(image_syn), dtype=bool)
            mask[args.treat_subject * args.ipc:(args.treat_subject + 1) * args.ipc] = False
            image_syn = image_syn[mask]
        else:
            assert args.selection_strategy in ['data_efficient_control',  'data_efficient_treat_label'] 
        return image_syn
        

    if args.teacher_label and args.dataset in ["CIFAR10", "CIFAR100", "Tiny", "ImageNet64", "BPTT"]: 
        # load pretrained experts for label generation
        expert_dir = os.path.join(args.expert_path, args.dataset)
        if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
            expert_dir = os.path.join(expert_dir+ "_NO_ZCA", 'ConvNet') 
        elif args.dataset in ["CIFAR10", "CIFAR100"] and args.zca:
            expert_dir = os.path.join(expert_dir, 'ConvNet') 
        elif args.dataset == "Tiny":
            expert_dir = os.path.join(expert_dir, 'imagenette', args.teacher_model) 
        elif args.dataset == "ImageNet64":
            expert_dir = os.path.join(expert_dir, 'imagenette', '64', args.teacher_model)
        elif args.dataset == "BPTT":
            expert_dir = os.path.join(args.expert_path, 'ImageNet64', 'imagenette', '64', args.teacher_model)
        else: 
            raise AssertionError("Unknown dataset")

        expert_files = []
        n = 1
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if len(expert_files) == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        else: 
            print("Expert Dir: {}".format(expert_dir), f"{n} buffers detected")

    def assign_softlabels():
        # reset expert and file index
        file_idx = 0
        expert_idx = 0

        label_net = get_network(args.teacher_model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        label_syn_ensemble = []
        while True: 
            if args.dataset in ["CIFAR10", "CIFAR100", "Tiny", "ImageNet64", "BPTT"]:
                print("loading file {}".format(expert_files[file_idx]))
                buffer = torch.load(expert_files[file_idx])
                expert_trajectory = buffer[expert_idx]
                # use the target param as the model param to get soft labels.
                target_params = expert_trajectory[args.max_expert_epoch]
            else: # ImageNet (cannot save/load all expert epochs in one file)
                expert_path = os.path.join(args.expert_path, f'model_{args.max_expert_epoch}.pth')
                target_params = torch.load(expert_path)

            if isinstance(target_params, list):
                label_net = ReparamModule(label_net)
                target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0).detach().requires_grad_(False)
            else:
                label_net.load_state_dict(target_params['model'])
            label_net.eval()

            batch_labels = []
            SOFT_INIT_BATCH_SIZE = 100
            if image_syn.shape[0] > SOFT_INIT_BATCH_SIZE:
                for indices in torch.split(torch.tensor([i for i in range(0, image_syn.shape[0])], dtype=torch.long), SOFT_INIT_BATCH_SIZE):
                    if isinstance(label_net, ReparamModule):
                        batch_labels.append(label_net(image_syn[indices].detach().to(args.device), flat_param=target_params))
                    else:
                        batch_labels.append(label_net(image_syn[indices].detach().to(args.device)).detach().cpu())
                label_syn = torch.cat(batch_labels, dim=0) 
            else:
                if isinstance(label_net, ReparamModule):
                    label_syn = label_net(image_syn.detach().to(args.device), flat_param=target_params)
                else:
                    label_syn = label_net(image_syn.detach().to(args.device))
            
            if not args.ensemble:
                label_syn = torch.nn.functional.softmax(label_syn * args.temp, dim=-1)
                break
            else: 
                if args.dataset == "ImageNet":
                    raise NotImplementedError("Ensemble not implemented for ImageNet")
                label_syn_ensemble.append(label_syn)
                # move to next expert if needed
                expert_idx += 1
                if expert_idx == len(buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        break
        if args.ensemble:
            label_syn = torch.stack(label_syn_ensemble).mean(dim=0)
            label_syn = torch.nn.functional.softmax(label_syn * args.temp, dim=-1)
        
        if args.selection_strategy == 'data_efficient_treat_label': # remove label (knowledge)
            # remove all information from that class
            label_syn_treated = label_syn.clone()
            label_syn_treated[:, args.treat_subject] = 0. 

            # # alternatively, generate random value for the treat subject
            # random_indices = torch.randint(0, label_syn_treated.shape[1], (label_syn_treated.shape[0],))
            # label_syn_treated[torch.arange(label_syn_treated.shape[0]), args.treat_subject] = label_syn_treated[torch.arange(label_syn_treated.shape[0]), random_indices]
            
            # only preserve softlabels for treat subject
            label_syn_treated[args.treat_subject * args.ipc:(args.treat_subject + 1) * args.ipc] = label_syn[args.treat_subject * args.ipc:(args.treat_subject + 1) * args.ipc]

            # re-normalize - if needed
            # label_syn_treated = label_syn_treated / label_syn_treated.sum(dim=1, keepdim=True)
            
            del label_syn
            label_syn = label_syn_treated

        if args.expt_type == 'ablation_k': # label swap experiment
            # get the largest element
            top_k_val, top_k_indices = torch.topk(label_syn, ablation_k)
            top_k_indices = top_k_indices[:, -1]
            top_k_val = top_k_val[:, -1]

            # get the smallest element
            bottom_k_val, bottom_k_indices = torch.topk(label_syn, 1, largest=False)
            bottom_k_indices = bottom_k_indices.squeeze()
            bottom_k_val = bottom_k_val.squeeze()

            # swap values 
            label_syn[list(range(len(label_syn))), top_k_indices] = bottom_k_val
            label_syn[list(range(len(label_syn))), bottom_k_indices] = top_k_val 

        if args.expt_type == 'data_knowledge_scaling':
            # only keep top k elements
            if args.label_topk > 0:
                label_syn_treated = label_syn.clone()
                top_k_val, top_k_indices = torch.topk(label_syn, args.label_topk, dim=1)
                label_syn_treated = torch.zeros_like(label_syn)
                label_syn_treated.scatter_(1, top_k_indices, top_k_val)

                del label_syn
                label_syn = label_syn_treated
                print(f"Lael syn with IPC={args.ipc}, keeping top {args.label_topk}")
            else:
                # use hardlabel
                label_syn = torch.tensor(np.array([np.ones(args.ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
                print(f"Lael syn with IPC={args.ipc}, using hard label")

        del label_net, target_params
        for _ in batch_labels:
            del _
        torch.cuda.empty_cache()
        # get_memory("softlabel assignment done")

        return label_syn.detach().requires_grad_(False)
                

    ''' generate experiment hyperparameters '''

    ''' optimal expert epoch'''
    if args.expt_type == 'tune_start': 
        tune_expert_epoch = np.arange(args.min_expert_epoch, args.max_expert_epoch+1, 2).tolist()
        expt_dict = {}
        for i, se in enumerate(tune_expert_epoch):
            expt_dict[i] = se
        print("Total number of experiments: {}".format(len(expt_dict)))
        print("start epoch list: {}".format(expt_dict))

   
        ''' temperature and start epoch grid '''
    elif args.expt_type == 'start_temp_grid':
        tune_expert_epoch = np.arange(args.min_expert_epoch , args.max_expert_epoch + 2, 6).tolist()
        tune_temp = 1.0 / np.linspace(1.0, 5.0, 10) # linear in temperature (not inverse of temperature)
        expt_dict = {}
        expt_num = 0
        for se in tune_expert_epoch:
            for t in tune_temp:
                expt_dict[expt_num] = (se, t)
                expt_num += 1
        print("Total number of experiments: {}".format(len(expt_dict)))
        print("grid: {}".format(expt_dict))
        
        ''' learning student rate '''
    elif args.expt_type == 'tune_lr':
        expt_dict = {}
        lr_list = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
        for i, lr in enumerate(lr_list):
            expt_dict[i] = lr
        print("Total number of experiments: {}".format(len(expt_dict)))
        print("lr list: {}".format(expt_dict))

        ''' ablation 1: importantce of k-th element in softmax'''
    elif args.expt_type == 'ablation_k':
        expt_dict = {}
        ablation_k_list = np.linspace(1, num_classes - 1 , num_classes - 1, dtype=int)
        for i, ablation_k in enumerate(ablation_k_list):
            expt_dict[i] = ablation_k
        print("Total number of experiments: {}".format(len(expt_dict)))
        print("ablation_k list: {}".format(expt_dict))

        ''' abalation 2: data efficiency '''
    elif args.expt_type in ['data_efficient_treat_label', 'data_efficient_treat_image','data_efficient_control']: 
        expt_dict = {}
        selection_strategy = [args.expt_type]
        subjec_idx = list(range(0, 40))
        exp_num = 0
        for i in subjec_idx:
            for s in selection_strategy: 
                expt_dict[exp_num] = (s, i)
                exp_num += 1
        print("Total number of experiments: {}".format(len(expt_dict)))
        print("data swap list: {}".format(expt_dict))
    
    elif args.expt_type == 'data_knowledge_scaling':
        expt_dict = {}
        # for Tiny ImageNet Expert epoch 11 experiment
        ipc_search_list = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 40][::-1]
        label_topk_list = ([2**x for x in range(8)] + [200] + [0])
        
        # for Tiny ImageNet Expert epoch 20 experiment
        # ipc_search_list = [1, 2, 5, 10, 15, 20, 30, 40, 50, 80, 100][::-1]

        # for Tiny ImageNet Expert epoch 50 experiment
        # ipc_search_list = [5, 10, 20, 30, 50, 100, 200][::-1]

        # for hard label basline experiment
        # ipc_search_list = [200][::-1]
        # label_topk_list = [0]

        exp_num = 0
        for k in label_topk_list:
            for i in ipc_search_list:
                expt_dict[exp_num] = (i, k)
                exp_num += 1
        print("Total number of experiments: {}".format(len(expt_dict)))
        print("Data-knowledge scaling search list: {}".format(expt_dict))

    elif args.expt_type == 'nothing':
        expt_dict = {0: None}
    else:
        raise AssertionError("Unknown experiment type")
            
    print('%s training begins'%get_time())
    
    best_acc = {m: 0 for m in [args.student_model]}

    best_std = {m: 0 for m in [args.student_model]}

    for exp_num in range(0, len(expt_dict)):
        # save_this_it = False
        args.exp_num = exp_num
        
        ''' assign hparams'''
        ''' expert epoch '''
        if args.expt_type == 'tune_start':
            args.max_expert_epoch = expt_dict[exp_num]

            ''' start epoch and temp grid'''
        elif args.expt_type == 'start_temp_grid':
            args.max_expert_epoch, args.temp = expt_dict[exp_num]

            ''' lr '''
        elif args.expt_type == 'tune_lr':
            args.lr_net = expt_dict[exp_num]

            ''' analysis 0: k element'''
        elif args.expt_type == 'ablation_k':
            ablation_k = expt_dict[exp_num]

            ''' analysis 2: data efficiency '''
        elif args.expt_type in ['data_efficiency', 'data_efficient_treat_label', 'data_efficient_treat_image', 'data_efficient_control']:
            args.selection_strategy, args.selection_idx = expt_dict[exp_num]
            image_syn = reassign_image_syn()

        elif args.expt_type == 'data_knowledge_scaling':
            # reassign image_syn
            del image_syn
            args.ipc, args.label_topk = expt_dict[exp_num]
            image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data

        else:
            pass
    

        print('-------------------------\nExperiment ID = %d'%exp_num)
        
        wandb.log({'Experiment': exp_num,
                   'Expert_Epoch': args.max_expert_epoch,
                   'Temp': args.temp,})
        
        if args.teacher_label:
            label_syn = assign_softlabels()
            # save softlabels for analysis
            # torch.save(label_syn, f'entropy/{args.dataset}_{args.selection_strategy}_ipc{args.ipc}_epoch{args.max_expert_epoch}.pt')

        else: # hard label from data
            if 'label_syn' not in locals(): # otherwise already loaded during the checkpoint
                label_syn = torch.tensor(np.array([np.ones(args.ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        ''' Evaluate '''
        model_eval = args.student_model
        print('-------------------------\nEvaluation\nExperiment ID = %d, model_train = %s, model_eval = %s'%(args.exp_num, args.teacher_model, model_eval))
        if args.dsa:
            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        else:
            print('DC augmentation parameters: \n', args.dc_aug_param)

        accs_test = []
        accs_train = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, dist=True)  #.to(args.device) # get a random model

            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn, label_syn, testloader, args)
            accs_test.append(acc_test)
            accs_train.append(acc_train)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std

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

    # basic settings
    parser.add_argument('--dataset', type=str, default='Tiny', help='dataset')
    parser.add_argument('--data_path', type=str, default='./', help='dataset path')
    parser.add_argument('--expert_path', type=str, default='.', help='expert path')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--teacher_model', type=str, default='ConvNet', help='model')
    parser.add_argument('--student_model', type=str, default='S',help='student model architecture, check utils.py for more info')

    # student training settings
    parser.add_argument('--epoch_eval_train', type=int, default=3000, help='epochs to train student model')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=["SGD", "AdamW"], help='optmizer to train a model with distilled data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='student learning rate')
    
    # experiment settings
    parser.add_argument('--expt_type', type=str, default=None, choices=['tune_start',  'tune_lr',  # tune two basic hpram: expert epoch and student lr
                                                                        'start_temp_grid', 'ablation_k', # Section 4 analysis
                                                                        'data_efficient_treat_label','data_efficient_treat_image', 'data_efficient_control', 
                                                                        'data_knowledge_scaling',
                                                                        'nothing'],
                                                                        help='which experiment to run, must specify')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for train data')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for test data evaluation')
    
    # image settings
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="load pretrained")
    parser.add_argument('--selection_strategy', type=str, default='random', choices=["random", 
                                                                                     "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "stratefied",
                                                                                     'data_efficient_treat_label','data_efficient_treat_image', 'data_efficient_control'],
                                                                                     help='how to select images from data')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    
    # expert label settings
    parser.add_argument('--teacher_label', action='store_true', default=False, help='whether to use expert generated soft label.')
    parser.add_argument('--max_expert_epoch', type=int, default=25, help='max epoch for expert epoch tuning')
    parser.add_argument('--min_expert_epoch', type=int, default=1, help='min epoch for expert epoch tuning')
    parser.add_argument('--ensemble', action='store_true', help="do expert ensemble for label generation")
    parser.add_argument('--temp', type=float, default=1.0, help='teacher label temp')


    # wandb settings
    parser.add_argument('--wandb', action='store_true', help="logging")
    parser.add_argument('--run_name', type=str, default=None, help='wandb expt name')
    parser.add_argument('--notes', type=str, default="No description", help='wandb additional notes')
    
    args = parser.parse_args()

    main(args)



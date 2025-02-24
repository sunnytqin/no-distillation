# train ImageNet
torchrun --nproc_per_node=1  train.py --model=resnet18 --data-path=/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet256 -b=256 --lr=0.0005 --output-dir=/n/holylabs/LABS/dam_lab/Lab/sqin/softlabels/ResNet50_tiny_lr  --print-freq=200

## CIFAR-100
python nodistill.py --dataset=CIFAR100 --ipc=50 --expt_type=nothing  --teacher_label  --max_expert_epoch=104 --lr_net=1.e-02  --expert_path=/n/holylabs/LABS/dam_lab/Lab/sqin/softlabels/results_100_F  --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/cifar100  --student_model=ConvNet --teacher_model=ConvNet --epoch_eval_train 3000 

## CIFAR 10
python nodistill.py --dataset=CIFAR10 --zca --ipc=1 --expt_type=nothing  --teacher_label  --max_expert_epoch=20 --lr_net=1.e-02  --expert_path=/n/holylabs/LABS/dam_lab/Lab/sqin/softlabels/results_100_F  --data_path=../data  --student_model=ConvNet --teacher_model=ConvNet --epoch_eval_train 3000 


# TinyImageNet
python nodistill.py --dataset=Tiny --ipc=1 --expt_type=nothing  --teacher_label  --max_expert_epoch=11 --lr_net=1.e-02  --expert_path=/n/holylabs/LABS/dam_lab/Lab/sqin/softlabels/results_100_F  --data_path=/n/home05/sqin/invariance/data/tiny-imagenet-200 --student_model=ConvNetD4 --teacher_model=ConvNetD4 --epoch_eval_train 1000

# ImageNet-1K
python nodistill.py --dataset=ImageNet --ipc=1 --lr_net=5.e-03 --expt_type=nothing --teacher_label --max_expert_epoch=17   --expert_path=/n/holylabs/LABS/dam_lab/Lab/sqin/softlabels/ResNet18_tiny_lr/   --data_path=/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet256  --student_model=ResNet18 --teacher_model ResNet18 --epoch_eval_train 3000 --optimizer=AdamW

## label swap test
python nodistill.py --dataset=Tiny --ipc=1 --expt_type=ablation_k --teacher_label --max_expert_epoch=7 --lr_net=1.e-02 --expert_path=/n/holyscratch01/dam_lab/sqin/softlabels/results_100_F --data_path=/n/home05/sqin/invariance/data/tiny-imagenet-200 --load_checkpoint=ipc1_random_image_syn.pt --num_eval=2 --student_model=ConvNetD4 --teacher_model=ConvNetD4 --epoch_eval_train=3000 --optimizer=SGD

## scaling law
python nodistill.py --dataset=Tiny --ipc=1 --expt_type=data_knowledge_scaling  --teacher_label --max_expert_epoch=7 --lr_net=1.e-02 --expert_path=/n/holyscratch01/dam_lab/sqin/softlabels/results_100_F --data_path=/n/home05/sqin/invariance/data/tiny-imagenet-200 --student_model=ConvNetD4 --teacher_model=ConvNetD4 --epoch_eval_train=3000 --optimizer=SGD 

## zero-shot experiment
python nodistill.py --dataset=Tiny --ipc=1 --expt_type=data_efficient_treat_image --teacher_label --max_expert_epoch=7 --lr_net=1.e-02 --expert_path=/n/holyscratch01/dam_lab/sqin/softlabels/results_100_F --data_path=/n/home05/sqin/invariance/data/tiny-imagenet-200 --load_checkpoint=ipc1_random_image_syn.pt --student_model=ConvNetD4 --teacher_model=ConvNetD4 --epoch_eval_train=3000 --optimizer=SGD

# train CIFAR10 expert 
 python buffer.py --dataset=CIFAR10 --model=ConvNet --zca  --train_epochs=400 --num_experts=1  --buffer_path=/n/holylabs/LABS/dam_lab/Lab/sqin/softlabels/results_100_F  --data_path=../data --save_interval 1 --lr_teacher=1e-3

# train CIFAR100 expert
python buffer.py --dataset=CIFAR100 --model=ConvNet  --train_epochs=150 --num_experts=1  --buffer_path=/n/holylabs/LABS/dam_lab/Lab/sqin/softlabels/results_100_F  --data_path=/n/holylabs/LABS/dam_lab/Lab/sqin/data --save_interval 1 --lr_teacher=1e-3

<!-- 
# All experts can be found at:
/n/holylabs/LABS/dam_lab/Lab/sqin/softlabels -->
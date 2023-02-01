""" Test Genotypes """
import sys

# TODO you need to change the PATH to your PATH.
path = "/home/yangqi/workspace/research_work/Index_work/cdarts_index_work"
if path not in sys.path:
    sys.path.insert(0, path)
import os
import argparse
import torch
import json
import time
import numpy as np
from torchvision import datasets
from models.model_test import ModelTest
import utils.utils as utils
from utils.Logger import Logger
from core.augment_function import validate
from data.cifar import _data_transforms_cifar

def main():

    parser = argparse.ArgumentParser(description="CDARTs Test Training With PyTorch")
    parser.add_argument("--model_name", default="CDARTs-Test", type=str, help="The model name")
    parser.add_argument("--model_arch", default="Gumbel_model_M4", help="Geno type architecture.", type=str)
    parser.add_argument("--model_path", default="Gumbel_model_M4", help="The test model path", type=str)
    ########### basic settings ############
    parser.add_argument("--device_gpu", default="2", type=str, help="Cuda device, i.e. 0 or 0,1,2,3")
    parser.add_argument("--data_config", default="configs/CIFAR10.yaml", metavar="FILE", help="path to data cfg file", type=str)
    parser.add_argument("--save", type=str, default="/data/yangqi/checkpoints_save/index_work_CDARTs/checkpoints/", help="Save model checkpoints in the specified directory")
    parser.add_argument('--param_pool_path', type=str, default=None, help='')
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--stem_multiplier', type=int, default=3)   #?这是啥？
    parser.add_argument('--report_period', type=int, default=100, help='print frequency')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') #! Default is 128
    parser.add_argument("--clip_grad_norm", type=float, default=5., help="gradient clipping for weights") 
    parser.add_argument('--init_channels', type=int, default=36)
    parser.add_argument('--layers', type=int, default=20, help='# of layers')
    parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
    parser.add_argument('--cutout', action='store_true', default=True, help='Whether to use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--sample_archs', type=int, default=1, help='sample arch num')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path prob')

    ########### alternate training ############
    parser.add_argument('--res_stem', action='store_true', default=False, help='resnet stem(pretrain)')
    parser.add_argument('--layer_num', type=int, default=3, help='layer need to be replaced')
    parser.add_argument('--cells_num', type=int, default=3, help='cells num of one layer')
    parser.add_argument('--same_structure', action='store_true', default=False, help='same_structure search and retrain')
    parser.add_argument('--ensemble_sum', action='store_true', default=False, help='whether to ensemble')
    parser.add_argument('--ensemble_param', action='store_true', default=False, help='whether to learn ensemble params')
    parser.add_argument('--use_beta', action='store_true', default=False, help='whether to use beta arch param')
    parser.add_argument('--bn_affine', action='store_true', default=False, help='main bn affine')
    parser.add_argument('--repeat_cell', action='store_true', default=False, help='use repeat cell')
    parser.add_argument('--fix_head', action='store_true', default=False, help='whether to fix head')
    parser.add_argument('--share_fc', action='store_true', default=False, help='whether to share fc')
    parser.add_argument('--sample_pretrain', action='store_true', default=False, help='sample_pretrain')
    parser.add_argument('--mixup_alpha', default=0., type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--resume_name', type=str, default='retrain_resume.pth.tar')


   
    # others
    parser.add_argument("--amp", action="store_true", default=False, help="Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.")
    args = parser.parse_args()
    # Logger
    os.makedirs(
        "logs/{}/".format(args.model_name), exist_ok=True,
    )
    log_path = "{}_{}_{}".format(args.model_name, args.data.NAME, time.strftime("%Y%m%d-%H"),)
    log = Logger("logs/{}/".format(args.model_name) + log_path + ".log", level="debug",)
    if args.local_rank == 0:
        log.logger.info("gpu device = %s" % args.device_gpu)
        log.logger.info(
            "args = %s", args,
        )
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    # Random seed
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)    
    
    # Load dataset     
    if args.data.NAME == "CIFAR10":    
        train_transform, valid_transform = _data_transforms_cifar(args)
        valid_dataset =  datasets.CIFAR10(root=args.data.DATASET_PATH, train=False, download=True, transform=valid_transform)
    elif args.data.NAME == "imagenet":
        # TODO
        return    
        # elif 'imagenet' in args.dataset:
        #     from lib.datasets.imagenet import get_augment_datasets
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    # Load Genotype
    genotype = eval("genotypes.%s" % args.model_arch) #! 我们这里三层的genotype都是一样的[3,3,2],碰到不一样的再说！
    if args.local_rank == 0:  
        log.logger.info(genotype) 
   
    model_main = ModelTest(genotype, args.model_type, args.res_stem, init_channel=args.init_channels, stem_multiplier=args.stem_multiplier, n_nodes=4, num_classes=args.n_classes,)
    model_path = torch.load(args.model_path, map_location="cpu",)
    model_main.load_state_dict(
        resume_state, strict=False,
    )
    model_main = model_main.cuda()

    writer = None
    (top1, top5,) = validate(valid_dataloader, model_main, 0, 0, writer, log, args)
    if args.local_rank == 0:
        print("Final best Prec@1 = {:.4%}, Prec@5 = {:.4%}".format(top1, top5,))


if __name__ == "__main__":
    main()

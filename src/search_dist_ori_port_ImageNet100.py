import os
import time
import logging
import argparse
import moxing as mox
#mox.file.shift('os', 'mox')
import sys
from datasets.dataset_config import dataset_config

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    # added bu jianzhong
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--nproc_per_node', default=8, type=int, help='nproc_per_node')
    parser.add_argument('--data_url', default='', type=str, help='data dir')
    parser.add_argument('--train_url', help='trainlog url.',type=str)
    parser.add_argument('--tmp_data_dir', help='dataset url.',type=str,default='/cache/imagenet')
    parser.add_argument('-loadd','--load_data',type=str2bool,default=True)
    parser.add_argument('--init_method', default='tcp://$MA_CURRENT_IP:6666', type=str, help='init_method')
    # parser.add_argument('--config', default='./config/finetune/deit_base_cifar100_adamw.yaml', type=str, help='init_method')
    # parser.add_argument('--data_path', default='s3://bucket-6643/niexing/data/cifar100', type=str, help='data path')
    # parser.add_argument('--pretrained_dir', default='s3://bucket-6643/niexing/code/VPT/Transformer/pretrained', type=str, help='data path')
    # parser.add_argument('--pretrained_model', default='deit_base_patch16_224-b5f2ef4d.pth', type=str, help='data path')
    parser.add_argument('--l-alpha', default=1.0, type=float, required=True,
                        help='The weight of the 1-rd branch in loss.')
    parser.add_argument('--l-beta', default=1.0, type=float, required=True,
                        help='The weight of the 2-rd branch in loss.')
    parser.add_argument('--l-gamma', default=1.0, type=float, required=True,
                        help='The weight of the distill loss.')
    parser.add_argument('--num-tasks', default=6, type=int, help='num-tasks')
    parser.add_argument('--datasets', default=['cifar100_icarl'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")

    # Mox args
    parser.add_argument('--mox_data_url', default='/cache/dataset', type=str, help='mox data path')
    parser.add_argument('--port_set', default=6666, type=int, help='port_set')

    #
    args, unparsed = parser.parse_known_args()

    os.system('nvidia-smi')
    # mox.file.make_dirs('/cache/rwightman')
    mox.file.make_dirs('/cache/trans_prompt')
    # mox.file.copy_parallel('s3://bucket-6643/niexing/code/VPT/Transformer_code/dyna_final', '/cache/trans_prompt')
    mox.file.copy_parallel('s3://bucket-6643/niexing/code/Continuous_learning/Lucir_Decom_Distill_detach_branch1_first_init_allres/src', '/cache/trans_prompt')
    
    '''
    try:
        strs = 'python -m pip install torch==1.11.0 torchvision==0.12.0' 
        os.system(strs)
        strs = 'python -m pip install timm==0.5.4' 
        os.system(strs)
        strs = 'python -m pip install opencv-python==4.4.0.46'
        os.system(strs)
        strs = 'python -m pip install termcolor==1.1.0' 
        os.system(strs)
        strs = 'python -m pip install yacs==0.1.8' 
        os.system(strs)
        strs = 'python -m pip install scipy' 
        os.system(strs)
        strs = 'python -m pip install ptflops' 
        os.system(strs)
    except:
        print('Installing failed!')
        #return
        

    if args.load_data: 
        if not os.path.exists('/cache/data'):
            start_time = time.time()
            print('copy data to cache ...')
            mox.file.make_dirs('/cache/data')
            
            mox.file.copy_parallel('{}'.format(args.data_path), '/cache/data')
            end_time = time.time()
            duration = end_time - start_time
            logging.info('Processing data time: {}s'.format(duration))


    if not os.path.exists('/cache/pretrained'):
        start_time = time.time()
        print('copy pretrained model to cache ...')
        mox.file.make_dirs('/cache/pretrained')
        
        mox.file.copy_parallel('{}'.format(args.pretrained_dir), '/cache/pretrained')
        end_time = time.time()
        duration = end_time - start_time
        logging.info('Processing pretrained model time: {}s'.format(duration))
    '''

    #-----------------------------------------------------------------------------------------------------------------
    # Add Result mox by NieX.
    if not os.path.exists('/cache/results'):
        print('create /cache/results ...')
        mox.file.make_dirs('/cache/results')
    #-----------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------
    # Add Dataset mox by NieX.
    if not os.path.exists(args.mox_data_url):
        mox.file.make_dirs(args.mox_data_url)
    if args.datasets == ['imagenet_100'] or args.datasets == ['imagenet_1000']:
        print('Copy imagenet to cache ...')
        if not mox.file.exists(os.path.join(args.mox_data_url, 'imagenet')):
            mox.file.copy(os.path.join(args.data_url, 'imagenet.tar'), os.path.join(args.mox_data_url, 'imagenet') + '.tar')
            cmd = 'cd /cache/dataset && tar -xf /cache/dataset/imagenet.tar'
            os.system(cmd)
            mox.file.copy(os.path.join(args.data_url, 'train_100.txt'), os.path.join(args.mox_data_url, 'imagenet/train_100.txt'))
            mox.file.copy(os.path.join(args.data_url, 'val_100.txt'), os.path.join(args.mox_data_url, 'imagenet/val_100.txt'))
            mox.file.copy(os.path.join(args.data_url, 'train_1000.txt'), os.path.join(args.mox_data_url, 'imagenet/train_1000.txt'))
            mox.file.copy(os.path.join(args.data_url, 'val_1000.txt'), os.path.join(args.mox_data_url, 'imagenet/val_1000.txt'))
            #if not mox.file.exists(os.path.join(args.mox_data_url, 'imagenet/train_100')):
                #mox.file.copy(os.path.join(args.data_url, 'train_100.txt'), os.path.join(args.mox_data_url, 'imagenet/train_100') + '.txt')
            # mox.file.copy_parallel(os.path.join(args.data_url, 'train_100.txt'), args.mox_data_url)
        else:
            mox.file.copy_parallel(args.data_url, args.mox_data_url)
    elif args.datasets == ['cifar100_icarl']:
        print('Copy cifar100 to cache ...')
        mox.file.copy_parallel('{}'.format(args.data_url), args.mox_data_url)
    else:
        print('Copy failed. Dataset type is wrong!')
    #-----------------------------------------------------------------------------------------------------------------

    # strs = ('cd /cache/rwightman/ && chmod +x distributed_search_ori_port_ImageNet100.sh')
    strs = ('cd /cache/trans_prompt/ && chmod +x distributed_search_ori_port_ImageNet100.sh')
    os.system(strs)
    logging.info(sys.argv)
    if args.world_size > 1:
        sys.argv.__delitem__(3)
        sys.argv.__delitem__(3)
    logging.info(sys.argv)
    # with open('/cache/rwightman/distributed_search_ori_port_ImageNet100.sh', 'r') as f:
    with open('/cache/trans_prompt/distributed_search_ori_port_ImageNet100.sh', 'r') as f:
        print(f.read())

    host = os.environ["MA_CURRENT_IP"]
    port = args.port_set
    # print(host)

    # print(host, port)
    
    #strs = 'cd /cache/rwightman/ && ./distributed_search_ori_port_ImageNet100.sh {} {} {} {} {}'.format(args.nproc_per_node,
    #                                                    args.world_size,
    #                                                    args.rank,
    #                                                    host, port,' '.join(sys.argv[1:]))
    
    strs = 'cd /cache/trans_prompt/ && ./distributed_search_ori_port_ImageNet100.sh {} {} {} {} {} {} {} {} {}'.format(args.nproc_per_node,
                                                        args.world_size,
                                                        args.rank,
                                                        host,
                                                        port,
                                                        args.l_alpha,
                                                        args.l_beta,
                                                        args.l_gamma,
                                                        args.num_tasks,
                                                        )
    print(strs)
    os.system(strs)

    # --------------------------------------------------------------------------------------------------------------
    # Add by NieX.
    print('Copy results to S3 ...')
    mox.file.copy_parallel('/cache/results', args.train_url)
    # --------------------------------------------------------------------------------------------------------------
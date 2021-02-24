#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import logging
from dataset import CheXpert, ChestRay, get_data_loader
from fixmatch import FixMatch
from loss import BCEWithLogitsLoss, get_category_list
import moco.loader
from ignite.contrib.metrics import ROC_AUC
from train_utils import get_logger,TBLog, get_SGD,get_cosine_schedule_with_warmup
# import ipdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--dataset',default='CheXpert', type=str, help="choose which dataset is trained")

parser.add_argument('--store-path', type=str, default='output/', help="path to store checkpoint and best model")

parser.add_argument('-f', '--finetune',dest='finetune',
                    help='unfreeze weight layers')

parser.add_argument('--fixmatch',action='store_true', help="use fixmatch to finetune the model and treat the trainset as unlabeled")

parser.add_argument('--uratio',type=float,default = 9.0,help="the number of unlabeled data to labeled data,9 is for 10% traindata")

parser.add_argument('--num_train_iter', type=int, default=2**20, 
                        help='total number of training iterations')

parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    
parser.add_argument('--num_workers', type=int, default=1)

parser.add_argument('--hard_label', type=bool, default=True)

parser.add_argument('--T', type=float, default=0.5)

parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')

parser.add_argument('--p_cutoff_pos', default=[0.95,0.95,0.95,0.95,0.95], nargs='*', type=float,
                    help='positive cutoff value for five classes')

parser.add_argument('--p_cutoff_neg', default=[0.2,0.2,0.2,0.2,0.2], nargs='*', type=float,
                    help='negative cutoff value for five classes')

parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

parser.add_argument('--num_eval_iter', type=int, default=10000,
                        help='evaluation frequency')

parser.add_argument('--amp', action='store_true', help='use mixed precision training or not')

parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

best_acc1 = 0

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    # model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
    #                             padding=3, bias=False)

    # freeze all layers but the last fc
    if not args.finetune:
        for name, param in model.named_parameters():
            if name not in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = False
                
    # modify the fc layer
    input_num = model.classifier.in_features
    # TODO: here's a hard code num_classes
    if args.dataset == 'CheXpert':
        model.classifier = nn.Linear(input_num,5,True)
    elif args.dataset == 'ChestRay':
        model.classifier = nn.Linear(input_num,14,True)
    model.classifier.weight.data.normal_(mean=0.0, std=0.01)
    model.classifier.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.classifier'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_auc']
            if args.gpu is not None and type(best_acc1) != float:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.5],
                                     std=[0.5])
    train_augmentation = [
            transforms.RandomResizedCrop(320, scale=(0.08, 1.0),ratio=(0.75, 1.333333333)),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            moco.loader.Convert2RGB(),
            moco.loader.EqualizeHist(),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    # valid dataset
    valid_augmentation = [
            moco.loader.Convert2RGB(),
            transforms.CenterCrop(320),
            moco.loader.EqualizeHist(),
            # moco.loader.GaussianBlur([.1, 2.]),
            transforms.ToTensor(),
            normalize
        ]

    # train fixmatch load labeled_data, unlabeled_data, eval_data 
    if args.fixmatch:

        #SET save_path and logger
        save_path = args.store_path
        logger_level = "WARNING"
        tb_log = None
        if args.rank % ngpus_per_node == 0:
            tb_log = TBLog(save_path, 'tensorboard')
            logger_level = "INFO"
        
        logger = get_logger('logs', save_path, logger_level)
        logger.warning(f"USE GPU: {args.gpu} for training")

        weak_augmentation = [
                transforms.RandomResizedCrop(320, scale=(0.08, 1.0),ratio=(0.75, 1.333333333)),
                moco.loader.Convert2RGB(),
                moco.loader.EqualizeHist(),
                transforms.ToTensor(),
                normalize
            ]
        
        labeled_train_dataset = eval(args.dataset)('train', transforms.Compose(train_augmentation),fixmatch = args.fixmatch,weak_transform = transforms.Compose(weak_augmentation))
        unlabeled_train_dataset = eval(args.dataset)('unlabeled', transforms.Compose(train_augmentation),fixmatch = args.fixmatch,weak_transform = transforms.Compose(weak_augmentation))
        test_dataset = eval(args.dataset)('valid', transforms.Compose(valid_augmentation))
        
        loader_dict = {}
        dset_dict = {'train_lb': labeled_train_dataset, 'train_ulb': unlabeled_train_dataset, 'eval': test_dataset}
        
        loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                                args.batch_size,
                                                data_sampler = args.train_sampler,
                                                num_iters=args.num_train_iter,
                                                num_workers=args.num_workers, 
                                                distributed=args.distributed)
        
        loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                                args.batch_size*args.uratio,
                                                data_sampler = args.train_sampler,
                                                num_iters=args.num_train_iter,
                                                num_workers=4*args.num_workers,
                                                distributed=args.distributed)
        
        loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                            args.eval_batch_size,
                                            num_workers=args.num_workers)
    
        num_classes = labeled_train_dataset.get_num_classes()
        annotations = labeled_train_dataset.get_annotations()
        labeled_num_class_list, _ = get_category_list(annotations,num_classes) 
        # define loss function (criterion) and optimizer
        labeled_criterion = BCEWithLogitsLoss(labeled_num_class_list).cuda(args.gpu)

        annotations = labeled_train_dataset.get_annotations()
        unlabeled_num_class_list, _ = get_category_list(annotations,num_classes) 
        # define loss function (criterion) and optimizer
        unlabeled_criterion = BCEWithLogitsLoss(unlabeled_num_class_list).cuda(args.gpu)
    
    # normal train load train_data,valid_data,test_data
    else:
        train_dataset = eval(args.dataset)('train', transforms.Compose(train_augmentation))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        
        val_dataset = eval(args.dataset)('valid', transforms.Compose(valid_augmentation))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        test_dataset = eval(args.dataset)('test', transforms.Compose(valid_augmentation))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        num_classes = train_dataset.get_num_classes()
        annotations = train_dataset.get_annotations()
        num_class_list, _ = get_category_list(annotations,num_classes) 
        # define loss function (criterion) and optimizer
        criterion = BCEWithLogitsLoss(num_class_list).cuda(args.gpu)
        if args.evaluate:
            validate(test_loader, model, criterion, args, num_classes)
            return

    if not args.fixmatch:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, num_classes)

            # evaluate on validation set
            acc1,auc_list = validate(val_loader, model, criterion, args, num_classes)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                checkpoint_dict = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_auc': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }
                for i, auc in enumerate(auc_list):
                    checkpoint_dict['auc'+str(i)] = auc

                save_checkpoint(checkpoint_dict, is_best, args.store_path+'checkpoint.pth.tar')
                if epoch == args.start_epoch and not args.finetune:
                    sanity_check(model.state_dict(), args.pretrained)
    else:
        # SET FixMatch: class FixMatch in models.fixmatch
        # TODO: backbone bn_momentum should be set by ema_m
        # args.bn_momentum = 1.0 - args.ema_m
        
        # SET Optimizer & LR Scheduler
        ## construct SGD and cosine lr scheduler
        optimizer = get_SGD(model, 'SGD', args.lr, args.momentum, args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter*0)

        # ipdb.set_trace()
        fixmatch_model = FixMatch(model,
                        num_classes,
                        args.ema_m,
                        args.T,
                        args.p_cutoff_pos,
                        args.p_cutoff_neg,
                        args.ulb_loss_ratio,
                        labeled_criterion,
                        unlabeled_criterion,
                        args.hard_label,
                        num_eval_iter=args.num_eval_iter,
                        tb_log=tb_log,
                        logger=logger)

        ## set SGD and cosine lr on FixMatch 
        fixmatch_model.set_optimizer(optimizer, scheduler)

        ## set DataLoader on FixMatch
        fixmatch_model.set_data_loader(loader_dict)

        # START TRAINING of FixMatch
        trainer = fixmatch_model.train
        for epoch in range(args.epochs):
            trainer(args, logger=logger)
            
        if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            fixmatch_model.save_model('latest_model.pth', args.store_path)
            
        logging.warning(f"GPU {args.rank} training is FINISHED")



def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def train(train_loader, model, criterion, optimizer, epoch, args, num_classes):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
   
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    roc_auc = ROC_AUC(activated_output_transform)
    roc_auc.reset()
    roc_auc_list = []
    for i in range(num_classes):
        roc_auc_list.append(ROC_AUC(activated_output_transform))
        roc_auc_list[-1].reset()

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(args, output, target)

        roc_auc.update((output.data,target.data))
        assert output.shape[1] ==num_classes,f'{output.shape}'
        for j in range(num_classes):
            roc_auc_list[j].update((output.data[:,j],target.data[:,j]))

        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    pbar_str = "---Epoch:{:>3d}    Epoch_Auc:{:>5.5f}---".format(
            epoch, roc_auc.compute()
    )
    print(pbar_str)
    for j in range(num_classes):
        pbar_str = "-------Epoch:{:>3d}  Category_id:{:>3d}   Valid_Auc:{:>5.5f}-------".format(
            epoch, j, roc_auc_list[j].compute()
        )
        print(pbar_str)


def validate(val_loader, model, criterion, args,num_classes):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    roc_auc = ROC_AUC(activated_output_transform)
    roc_auc.reset()
    roc_auc_list = []
    for i in range(num_classes):
        roc_auc_list.append(ROC_AUC(activated_output_transform))
        roc_auc_list[-1].reset()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(args, output, target)

            # measure accuracy and record loss
            roc_auc.update((output.data,target.data))
            for j in range(num_classes):
                roc_auc_list[j].update((output.data[:,j],target.data[:,j]))
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        auc_list = []
        pbar_str = "---Valid--Epoch_Auc:{:>5.5f}---".format(
            roc_auc.compute()
        )
        print(pbar_str)
        for j in range(num_classes):
            auc_temp = roc_auc_list[j].compute()
            auc_list.append(auc_temp)
            pbar_str = "-------Valid--Category_id:{:>3d}   Valid_Auc:{:>5.5f}-------".format(
                j, auc_temp
            )
            print(pbar_str)

    return roc_auc.compute(),auc_list


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/'.join(filename.split('/')[:-1])+'/model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'classifier.weight' in k or 'classifier.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

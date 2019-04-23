"""Training file of the original ResNet model"""
import argparse
import shutil
import time

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import logging
import models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'])
    parser.add_argument('--data', type=str, default='/ssd2/bansa01/imagenet_final/',
                        help='path to dataset')
    parser.add_argument('arch',  default='resnet101',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet74)')
    parser.add_argument('experiment', default='unknown', type=str,
                        help='the experiment name')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--save-folder', default='checkpoints', type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--lr-adjust', dest='lr_adjust',
                        choices=['linear', 'step'], default='step')
    parser.add_argument('--crop-size', dest='crop_size', type=int, default=224)
    parser.add_argument('--scale-size', dest='scale_size', type=int,
                        default=256)
    parser.add_argument('--step-ratio', dest='step_ratio', type=float,
                        default=0.1)
    parser.add_argument('--randInd', default=0.0, type=float,
                        help='the proportion of freezed params')
    parser.add_argument('--refreshFreeze', default=1, type=int,
                        help='the epoch number the refresh the target freezing layer')
    # parser.add_argument('--freezeBN', action='store_true',
    #                     help='If the batch norm is randomly freezed while training')
    parser.add_argument('--endFinetune', default=0, type=int,
                        help='the epochs for finetuning the whole network')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    args.device = device

    args.save_path = save_path = os.path.join(args.save_folder, args.experiment, args.arch)

    # first, detect four model saving folder
    if os.path.exists(save_path):
        print("the folder of this experiment already exists, Please check if it's safe to overwrite")
        return
    else:
        print("making directory for result saving")
        os.makedirs(save_path)

    # create logger with 'spam_application'
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(args.logger_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    if args.cmd == 'train':
        logger.info("start training {}".format(args.arch))
        run_training(args, logger)
    elif args.cmd == 'test':
        logger.info('start evaluating {} with checkpoints from {},'
                     .format(args.arch, args.resume))
        test_model(args, logger)


def run_training(args, logger):
    # create model
    model = models.__dict__[args.arch](args.pretrained, randInd=args.randInd)
    model = torch.nn.DataParallel(model).to(args.device)

    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(args.scale_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    finetuneEpoch = args.epochs - args.endFinetune
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch, logger)

        # train for one epoch

        # train(args, train_loader, model, criterion, optimizer, epoch, args.device, logger,
        #       refreshFreeze=(epoch % args.refreshFreeze == 0), finetuneFlag=(epoch >= finetuneEpoch))

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion, logger)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = os.path.join(args.save_path,
                                       'checkpoint_{:03d}.pth.tar'.format(
                                           epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path,
                        os.path.join(args.save_path, 'checkpoint_latest.pth.tar'))


def test_model(args, logger):
    # create model
    model = models.__dict__[args.arch](args.pretrained)

    model = torch.nn.DataParallel(model).to(args.device)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t = transforms.Compose([
        transforms.Scale(args.scale_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, t),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(args.device)

    validate(args, val_loader, model, criterion, logger)


def train(args, train_loader, model, criterion, optimizer, epoch, device, logger, refreshFreeze, finetuneFlag):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # compute output
    if finetuneFlag:
        for param in model.parameters():
            param.requires_grad = True

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # compute output
        model.module.refreshFreeze = (not finetuneFlag) and refreshFreeze
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(float(loss), input.size(0))
        top1.update(float(prec1), input.size(0))
        top5.update(float(prec5), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(args, val_loader, model, criterion, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    device = args.device
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)

            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(float(loss), input.size(0))
        top1.update(float(prec1), input.size(0))
        top5.update(float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(
                'Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def adjust_learning_rate(args, optimizer, epoch, logger):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    logger.info('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

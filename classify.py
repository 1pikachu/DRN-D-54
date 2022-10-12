import argparse
import shutil
import time

import numpy as np
import os
from os.path import exists, split, join, splitext

import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import drn as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'])
    parser.add_argument('--data', metavar='DIR', default=None,
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='drn18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: drn18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
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
    parser.add_argument('--check-freq', default=10, type=int,
                        metavar='N', help='checkpoint frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--lr-adjust', dest='lr_adjust',
                        choices=['linear', 'step'], default='step')
    parser.add_argument('--crop-size', dest='crop_size', type=int, default=224)
    parser.add_argument('--scale-size', dest='scale_size', type=int, default=256)
    parser.add_argument('--step-ratio', dest='step_ratio', type=float, default=0.1)
    # oob
    parser.add_argument('--device', choices=["cpu", "cuda", "xpu"], default="cpu", type=str)
    parser.add_argument('--dummy', action="store_true", default=True, help="use dummy input")
    parser.add_argument("--num_warmup", "--warmup_iter", type=int, default=20, help="The number warmup, default is 20.")
    parser.add_argument("--num_iters", "--early_stop_at_iter",type=int, default=200, help="The number iters of benchmark, default is 200.")
    parser.add_argument('--precision', choices=["float32", "float16", "bfloat16"], default='float32', help='Precision')
    parser.add_argument('--image_size', type=int, default=224, help="input img size")
    parser.add_argument("--jit", action="store_true", help="Use jit optimize to do optimization.")
    parser.add_argument("--nv_fuser", action="store_true")
    parser.add_argument("--channels_last", type=bool, default=False, help="Use pytorch NHWC.")
    parser.add_argument("--profile", action="store_true", default=False, help="Trigger profile on current topology.")
    parser.add_argument("--bn_folding", action="store_true", default=False)

    args = parser.parse_args()
    return args


def main():
    print(' '.join(sys.argv))
    args = parse_args()
    print(args)
    if args.device == "xpu":
        import intel_extension_for_pytorch

    if args.cmd == 'train':
        run_training(args)
    elif args.cmd == 'test':
        test_model(args)


def run_training(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)

    model = torch.nn.DataParallel(model).cuda()

    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        checkpoint_path = 'checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % args.check_freq == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)

    # model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if not args.dummy:
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
    else:
        val_loader = None

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()

    print("precision: ", args.precision)
    if args.precision == "bfloat16":
        print("---- bfloat16 autocast")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            validate(args, val_loader, model, criterion)
    elif args.precision == "float16" and args.device == "cuda":
        print("---- float16 autocast")
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            validate(args, val_loader, model, criterion)
    else:
        validate(args, val_loader, model, criterion)


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    total_time = 0
    total_count = 0
    # dummy input
    input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    # device
    input = input.to(args.device)
    model = model.to(args.device)
    # precision
    if args.device == "xpu" and args.precision == "float16":
        input = input.half()
        model = model.half()
        print("---- float16 to.half()")
    # channels_last
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        input = input.to(memory_format=torch.channels_last)
    # nv fuser
    if args.nv_fuser:
        fuser_mode = "fuser2"
    else:
        fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)
    # jit
    if args.jit:
        try:
            model = torch.jit.trace(model, input, check_trace=False)
            print("---- With JIT enabled.")
            if args.bn_folding:
                from torch.jit._recursive import wrap_cpp_module
                model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
                print("---- With bn folding")
            model = torch.jit.freeze(model)
        except (RuntimeError, TypeError) as e:
            print("---- With JIT disabled.")
            print("failed to use PyTorch jit mode due to: ", e)
    if args.profile and args.device == "xpu":
        for i in range(args.num_iters + args.num_warmup):
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                start_time = time.time()
                output = model(input)
                torch.xpu.synchronize()
            duration = time.time() - start_time
            print("Iteration: ", duration)
            if i >= args.num_warmup:
                total_time += duration
                total_count += 1
            if args.profile and i == int((args.num_iters + args.num_warmup)/2):
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=int((args.num_iters + args.num_warmup)/2),
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
            for i in range(args.num_iters + args.num_warmup):
                start_time = time.time()
                output = model(input)
                torch.cuda.synchronize()
                duration = time.time() - start_time
                # profile iter update
                p.step()
                print("Iteration: ", duration)
                if i >= args.num_warmup:
                    total_time += duration
                    total_count += 1
    elif not args.profile:
        for i in range(args.num_iters + args.num_warmup):
            start_time = time.time()
            output = model(input)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elif args.device == "cuda":
                torch.cuda.synchronize()
            duration = time.time() - start_time
            print("Iteration: ", duration)
            if i >= args.num_warmup:
                total_time += duration
                total_count += 1
    else:
        print("------please check params")
        return 1

    perf = args.batch_size * total_count / total_time
    print('inference Throughput: %3.3f fps'%perf)
    print('batch size: %d'%args.batch_size)
    print('device: %s'%args.device)


    return top1.avg


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
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

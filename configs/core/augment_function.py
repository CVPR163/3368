import torch
import torch.nn as nn
from utils import utils
from data import data_utils


def train(train_loader, model, optimizer, epoch,  scaler, log, args):
    device = torch.device("cuda")

    criterion = nn.CrossEntropyLoss(label_smoothing = args.label_smooth).to(device)


    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    step_num = len(train_loader)
    cur_step = epoch*step_num
    cur_lr = optimizer.param_groups[0]['lr']
    if args.local_rank == 0:  
        log.logger.info("Train Epoch {} LR {}".format(epoch, cur_lr))


    model.train()

    for step, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        
        input, target_a, target_b, lam = data_utils.mixup_data(input, target, args.mixup_alpha, use_cuda=True)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits, logits_aux = model(input)
            # loss = criterion(logits, target)
            loss = data_utils.mixup_criterion(criterion, logits, target_a, target_b, lam)
            if args.aux_weight > 0:
                # loss_aux = criterion(logits_aux, y)
                loss_aux = data_utils.mixup_criterion(criterion, logits_aux, target_a, target_b, lam)
                loss = loss + args.aux_weight * loss_aux
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        reduced_loss = loss.data
        N = input.size(0)
        losses.update(reduced_loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
 
        if step % args.report_period == 0 and args.local_rank == 0:
            log.logger.info("train setp:%03d Loss: %.4e Top1 acc: %.4f Top5 acc: %.4f", step, losses.avg, top1.avg, top5.avg)
 
       
    return top1.avg, losses.avg
def validate(valid_loader, model, cur_step, log, args):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()
    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

            reduced_loss = loss.data

            losses.update(reduced_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            
            torch.cuda.synchronize()
            step_num = len(valid_loader)

            if step % args.report_period == 0 and step > 0:
                log.logger.info("valid setp:%03d Loss: %.4e Top1 acc: %.4f Top5 acc: %.4f", step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg

import torch
import torch.nn as nn
from utils import utils
from models.loss import Loss_interactive

def search(train_loader, valid_loader, model, optimizer, w_optim, alpha_optim, layer_idx, epoch, writer, log, args):
    # interactive retrain and kl

    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    losses_interactive = utils.AverageMeter()
    losses_cls = utils.AverageMeter()
    losses_reg = utils.AverageMeter()

    step_num = len(train_loader)
    step_num = int(step_num * args.sample_ratio)

    cur_step = epoch*step_num
    cur_lr_search = w_optim.param_groups[0]['lr']
    cur_lr_main = optimizer.param_groups[0]['lr']
    if args.local_rank == 0:  
        log.logger.info("Train Epoch {} Search LR {}".format(epoch, cur_lr_search))
        log.logger.info("Train Epoch {} Main LR {}".format(epoch, cur_lr_main))
        if args.use_Tensorboard:
            writer.add_scalar('retrain/lr', cur_lr_search, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        if step > step_num:
            break

        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        #use valid data
        alpha_optim.zero_grad()
        optimizer.zero_grad()

        logits_search, emsemble_logits_search = model(val_X, layer_idx, super_flag=True)  #! Search Network  Cells:8  验证集，用于计算L_val^S 和 L_val^S,E
        logits_main, emsemble_logits_main= model(val_X, layer_idx, super_flag=False)   #! Nas layer: Evaluation Network  Top-k = 2 Cells:20 ， 用于计算L_val^E 和 L_val^S,E

        loss_cls = (criterion(logits_search, val_y) + criterion(logits_main, val_y)) / args.loss_alpha #! L_val^S and L_val^E 
        loss_interactive = Loss_interactive(emsemble_logits_search, emsemble_logits_main, args.loss_T, args.interactive_type) * args.loss_alpha #! L_val^S,E

        loss_regular = 0 * loss_cls  #! Reg 公式(11), 对DARTs的一个LOSS的优化
        if args.regular:
            reg_decay = max(args.regular_coeff * (1 - float(epoch-args.pretrain_epochs)/((args.search_iter-args.pretrain_epochs)*args.search_iter_epochs*args.regular_ratio)), 0)
            # normal cell
            op_opt = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect']
            op_groups = []
            for idx in range(layer_idx, 3):
                for op_dx in op_opt:
                    op_groups.append((idx - layer_idx, op_dx))
            loss_regular = loss_regular + model.add_alpha_regularization(op_groups, weight_decay=reg_decay, method='L1', reduce=False)

            # reduction cell
            
            op_opt = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect']
            op_groups = []
            for i in range(layer_idx, 3):
                for op_dx in op_opt:
                    op_groups.append((i - layer_idx, op_dx))
            loss_regular = loss_regular + model.add_alpha_regularization(op_groups, weight_decay=reg_decay, method='L1', normal=False)
                
 
        loss = loss_cls + loss_interactive + loss_regular  #! Stage 2 : Joint Learning Loss stage. loss_regular，在论文公式(11)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.w_grad_clip)
        optimizer.step()   #! update α and w_E
        alpha_optim.step()
                    
        prec1, prec5 = utils.accuracy(logits_main, val_y, topk=(1, 5))
        if args.distributed:
            reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
            reduced_loss_interactive = utils.reduce_tensor(loss_interactive.data, args.world_size)
            reduced_loss_cls = utils.reduce_tensor(loss_cls.data, args.world_size)
            reduced_loss_reg = utils.reduce_tensor(loss_regular.data, args.world_size)
            prec1 = utils.reduce_tensor(prec1, args.world_size)
            prec5 = utils.reduce_tensor(prec5, args.world_size)

        else:
            reduced_loss = loss.data
            reduced_loss_interactive = loss_interactive.data
            reduced_loss_cls = loss_cls.data
            reduced_loss_reg = loss_regular.data

        losses.update(reduced_loss.item(), N)
        losses_interactive.update(reduced_loss_interactive.item(), N)
        losses_cls.update(reduced_loss_cls.item(), N)
        losses_reg.update(reduced_loss_reg.item(), N)

        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        torch.cuda.synchronize()
        if args.local_rank == 0 and (step % args.print_freq == 0 or step == step_num):
            log.logger.info(
                "Train_2: Layer {}/{} Epoch {:2d}/{} Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Loss_interactive {losses_interactive.avg:.3f} Losses_cls {losses_cls.avg:.3f} Losses_reg {losses_reg.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    layer_idx+1, args.layer_num, epoch+1, args.search_iter*args.search_iter_epochs, step,
                    step_num, losses=losses, losses_interactive=losses_interactive, losses_cls=losses_cls,
                    losses_reg=losses_reg, top1=top1, top5=top5))

        if args.local_rank == 0 and args.use_Tensorboard:
            writer.add_scalar('retrain/loss', reduced_loss.item(), cur_step)
            writer.add_scalar('retrain/top1', prec1.item(), cur_step)
            writer.add_scalar('retrain/top5', prec5.item(), cur_step)
            cur_step += 1


        w_optim.zero_grad()
        logits_search_train, _ = model(trn_X, layer_idx, super_flag=True)  #! 用于计算 L_train^S   
        loss_cls_train = criterion(logits_search_train, trn_y)
        loss_train = loss_cls_train  #! L_train^S   
        loss_train.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), args.w_grad_clip)
        # only update w
        w_optim.step()  #! update W_S

        # alpha_optim.step()
        if args.distributed:
            reduced_loss_cls_train = utils.reduce_tensor(loss_cls_train.data, args.world_size)
            reduced_loss_train = utils.reduce_tensor(loss_train.data, args.world_size)
        else:
            reduced_loss_cls_train = reduced_loss_cls_train.data
            reduced_loss_train = reduced_loss_train.data

        if args.local_rank == 0 and (step % args.print_freq == 0 or step == step_num-1):
            log.logger.info(
                "Train_1: Loss_cls: {:.3f} Loss: {:.3f}".format(
                    reduced_loss_cls_train.item(), reduced_loss_train.item())
            )


    if args.local_rank == 0:  
        log.logger.info("Train_2: Layer {}/{} Epoch {:2d}/{} Final Prec@1 {:.4%}".format(
            layer_idx+1, args.layer_num, epoch+1, args.search_iter*args.search_iter_epochs, top1.avg))


def retrain_warmup(valid_loader, model, optimizer, layer_idx, epoch, writer, log, super_flag, retrain_epochs, args):

    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    step_num = len(valid_loader)
    step_num = int(step_num * args.sample_ratio)  #TODO 思考sample ratio=0.2，只从里面取0.2的比率来进行训练

    cur_step = epoch*step_num
    cur_lr = optimizer.param_groups[0]['lr']
    if args.local_rank == 0:  
        log.logger.info("Warmup Epoch {} LR {:.3f}".format(epoch+1, cur_lr))
        if args.use_Tensorboard:
            writer.add_scalar('warmup/lr', cur_lr, cur_step)

    model.train()

    for step, (val_X, val_y) in enumerate(valid_loader):
        if step > step_num:
            break

        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = val_X.size(0)

        optimizer.zero_grad()
        logits_main, _ = model(val_X, layer_idx, super_flag=super_flag)
        loss = criterion(logits_main, val_y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.w_grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits_main, val_y, topk=(1, 5))
        if args.distributed:
            reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
            prec1 = utils.reduce_tensor(prec1, args.world_size)
            prec5 = utils.reduce_tensor(prec5, args.world_size)

        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        torch.cuda.synchronize() #! 等待所有的CUDA程序运行完成
        if args.local_rank == 0 and (step % args.print_freq == 0 or step == step_num):
            log.logger.info(
                "Warmup: Layer {}/{} Epoch {:2d}/{} Step {:03d}/{:03d} Loss {losses.avg:.3f}  "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    layer_idx+1, args.layer_num, epoch+1, retrain_epochs, step,
                    step_num, losses=losses, top1=top1, top5=top5))

        if args.local_rank == 0 and args.use_Tensorboard == True:  
            writer.add_scalar('retrain/loss', reduced_loss.item(), cur_step)
            writer.add_scalar('retrain/top1', prec1.item(), cur_step)
            writer.add_scalar('retrain/top5', prec5.item(), cur_step)
            cur_step += 1

    if args.local_rank == 0:  
        log.logger.info("Warmup: Layer {}/{} Epoch {:2d}/{} Final Prec@1 {:.4%}".format(
            layer_idx+1, args.layer_num, epoch+1, retrain_epochs, top1.avg))

def validate(valid_loader, model, layer_idx, epoch, cur_step, writer, log, super_flag, args):
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

            logits, _ = model(X, layer_idx, super_flag=False)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

            reduced_loss = loss.data

            losses.update(reduced_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            
            torch.cuda.synchronize()
            step_num = len(valid_loader)

            if (step % args.print_freq == 0 or step == step_num-1) and args.local_rank == 0:
                log.logger.info(
                    "Valid: Layer {}/{} Epoch {:2d}/{} Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        layer_idx+1, args.layer_num, epoch+1, args.search_iter*args.search_iter_epochs, step, step_num,
                        losses=losses, top1=top1, top5=top5))

    if args.local_rank == 0 and args.use_Tensorboard:
        writer.add_scalar('val/loss', losses.avg, cur_step)
        writer.add_scalar('val/top1', top1.avg, cur_step)
        writer.add_scalar('val/top5', top5.avg, cur_step)

        log.logger.info("Valid: Layer {}/{} Epoch {:2d}/{} Final Prec@1 {:.4%}".format(
            layer_idx+1, args.layer_num, epoch+1, args.search_iter*args.search_iter_epochs, top1.avg))

    return top1.avg
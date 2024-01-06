import datetime
import torch
import time
import numpy as np
from Code.util import *
from Code.compute_auc import Auc_Data_Calc,AucResuluts_logs
import torch.nn.functional as F
from Code.focal_loss import FocalLoss
import os

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.acc = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FedEngine:
    def __init__(self, args, dataloader, model,global_param, server_param, local_param, outputs,cid, mode):
        self.args = args
        self.dataloader = dataloader

        self.global_param = global_param
        self.server_param = server_param
        self.local_param = local_param
        self.model = model
        model.load_state_dict(self.global_param)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args .lr, momentum=self.args.momentum, nesterov=True,
                                    weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-2)

        self.client_id = cid
        self.outputs = outputs
        self.mode = mode
        self.m1, self.m2, self.m3, self.reg1, self.reg2 = None, None, None, None, None


    def run(self):
        self.model.to(self.args.device)
        output = self.client_run()
        return output

    def client_run(self):
        if self.mode == "Train":
            # training process
            for epoch in range(self.args.client_epochs):
                acc,losses = self.batch_run(True,epoch)
                self.scheduler.step()

        elif self.mode == "Test":
            acc, losses = self.batch_run(False, 1)

        self.model.to("cpu")
        output = {"params": self.model.state_dict(),
                  "loss": losses.avg,
                  "acc": acc.avg*100,
                  "c_state": self.client_id}

        return output

    def batch_run(self, training,epoch):
        self.model.train(training)
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        end = time.time()
        lossfuction = FocalLoss(alpha=0.75, gamma=2, num_classes=2)
        preValue = []
        TrueLabel = []
        patienceNames = []
        G_serverName = ''

        for batch_idx,(image,label,name)in enumerate(self.dataloader):
            if self.args.CUDA == True:
                image = image.cuda()
                label = label.cuda()
            ServerName = name[0].split('\\')[-3]
            G_serverName = ServerName

            output,_ = self.model(image)
            loss = lossfuction(output, label)
            loss_sum = self.criterion(loss, self.mode)

            if training:
                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()
            batch_size = label.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)

            preds = F.softmax(output, dim=-1)[:, 1]
            preValue.append(preds.cpu().squeeze().detach().numpy())
            TrueLabel.append(label.cpu().squeeze().detach().numpy())
            patienceNames.append(name)

            acc.update(torch.eq(pred.squeeze(), label.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss_sum.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % 10 == 0 and training==True:
                res = '\t'.join([
                    'ServerName:%s' % (ServerName),
                    'Epoch: [%d/%d]' % (epoch , self.args.client_epochs),
                    'Iter: [%d/%d]' % (batch_idx + 1, len(self.dataloader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f' % (losses.avg),
                    'Acc %.4f ' % (acc.avg * 100),
                ])
                print(res)

        return acc,losses

    def criterion(self, loss, mode):
        if self.args.agg == "avg":
            pass
        elif mode != "PerTrain" :
            self.m1 = sd_matrixing(self.model.state_dict()).reshape(1, -1).to(self.args.device)
            self.m2 = sd_matrixing(self.server_param).reshape(1, -1).to(self.args.device)
            self.m3 = sd_matrixing(self.global_param).reshape(1, -1).to(self.args.device)
            self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
            self.reg2 = torch.nn.functional.pairwise_distance(self.m1, self.m3, p=2)
            loss = loss + self.args.serverbeta * (self.reg1+self.reg2)/(sum([p.nelement() for p in self.model.parameters()]))#统计参数个数
        return loss

    def Validation(self,training):
        self.model.to(self.args.device)
        self.model.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        end = time.time()
        lossfuction = FocalLoss(alpha=0.75, gamma=2, num_classes=2)
        preValue = []
        TrueLabel = []
        patienceNames = []
        G_serverName = ''

        for batch_idx,(image,label,name) in enumerate(self.dataloader):
            if self.args.CUDA == True:
                image = image.cuda()
                label = label.cuda()
            ServerName = name[0].split('\\')[-3]
            G_serverName = ServerName

            output,_ = self.model(image)
            #loss = torch.nn.functional.cross_entropy(output, label)
            loss = lossfuction(output, label)
            loss_sum = self.criterion(loss, self.mode)

            batch_size = label.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)

            preds = F.softmax(output, dim=-1)[:, 1]
            preValue.append(preds.cpu().squeeze().detach().numpy())
            TrueLabel.append(label.cpu().squeeze().detach().numpy())
            patienceNames.append(name)

            acc.update(torch.eq(pred.squeeze(), label.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss_sum.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0 and training=='Train':
                res = '\t'.join([
                    '%s_Valdation' % (training),
                    'ServerName:%s' % (ServerName),
                    'Iter: [%d/%d]' % (batch_idx + 1, len(self.dataloader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f ' % (losses.avg),
                    'Acc %.4f ' % (acc.avg * 100),
                ])
            else :
                res = '\t'.join([
                    '%s_Valdation'%(training),
                    'ServerName:%s' % (ServerName),
                    'Iter: [%d/%d]' % (batch_idx + 1, len(self.dataloader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f ' % (losses.avg),
                    'Acc %.4f ' % (acc.avg * 100),
                ])
            print(res)
#        TrainAuc = Auc_Data_Calc(preValue, TrueLabel, patienceNames)
#         print("%s ServerName:%s  PicturesAuc:%.4f PatientsAuc:%.4f" %
#               (training,G_serverName, TrainAuc[0], TrainAuc[1]))

        outputs = {"params": self.model.state_dict(),
                  "loss": losses.avg,
                  "acc": acc.avg * 100,
                  "c_state": G_serverName,
                  # "AucList":TrainAuc,
                  # "auc":TrainAuc[1]
                   }
        return outputs









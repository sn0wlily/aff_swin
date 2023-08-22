import torch
from core.data import build_dataloader
from core.model import build_model
from core.optimizer import build_optimizer, build_lr_scheduler
from core.loss import build_loss
from core.metric import build_metric
from utils.registery import SOLVER_REGISTRY
from utils.helper import format_print_dict
from utils.logger import get_logger_and_log_path
from utils.load import load_pretrained
from collections import OrderedDict
from torchvision import transforms
import os
import copy
import datetime
from torch.nn.parallel import DistributedDataParallel
import yaml
import numpy as np
import pandas as pd


@SOLVER_REGISTRY.register()
class RDropSolver(object):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.task = self.cfg['task']
        self.local_rank = torch.distributed.get_rank()
        self.train_loader, self.val_loader, self.train_weight = build_dataloader(cfg)
        self.len_train_loader, self.len_val_loader = len(self.train_loader), len(self.val_loader)
        # self.train_loader, self.train_weight = build_train_dataloader(cfg)
        # self.len_train_loader = len(self.train_loader)
        # if self.local_rank == 0:
        #     self.val_loader = build_val_dataloader(cfg)
        #     self.len_val_loader = len(self.val_loader)
        # self.transform = transforms.Compose([transforms.ToPILImage(),
        #                               transforms.Resize(size=(224, 224)),
        #                             #   transforms.RandomHorizontalFlip(),
        #                               # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        #                               transforms.ToTensor(),
        #                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #                               # transforms.RandomErasing(),
        #                               ])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(build_model(cfg))
        self.model_without_ddp = model
        self.model = DistributedDataParallel(model.cuda(self.local_rank), device_ids=[
                                             self.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        self.criterion = build_loss(cfg, self.train_weight).cuda(self.local_rank)
        self.optimizer = build_optimizer(cfg)(self.model.parameters(), **cfg['solver']['optimizer']['args'])
        self.hyper_params = cfg['solver']['args']
        crt_date = datetime.date.today().strftime('%Y-%m-%d')
        self.logger, self.log_path = get_logger_and_log_path(crt_date=crt_date, **cfg['solver']['logger'])
        self.metric_fn = build_metric(cfg)
        try:
            self.epoch = self.hyper_params['epoch']
        except Exception:
            raise 'should contain epoch in {solver.args}'
        if self.local_rank == 0:
            self.save_dict_to_yaml(self.cfg, os.path.join(self.log_path, 'config.yaml'))
            self.logger.info(self.cfg)
        if 'mix' in self.cfg.keys() and self.cfg['mix'] == True:
            self.epoch_prefix = True
        else:
            self.epoch_prefix = False
            
            
    def train(self):
        if torch.distributed.get_rank() == 0:
            self.logger.info('==> Start Training')
            
        if self.cfg['model']['pretrained']:
            new_state_dict = OrderedDict()
            model_state_dict = self.model_without_ddp.state_dict()
            checkpoint = torch.load(self.cfg['model']['pretrained'], map_location=torch.device('cpu'))
            for k, v in checkpoint['state_dict'].items():
                if 'backbone' in k:
                    name = k[9:]
                    # if v.size() == model_state_dict[name].size():
                    new_state_dict[name] = v
            self.model_without_ddp.load_state_dict(new_state_dict, strict=False)  
        # current_model_dict = model.state_dict()
        # loaded_state_dict = torch.load(path, map_location=torch.device('cpu'))
        # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
        # model.load_state_dict(new_state_dict, strict=False)
        
        # self.model_without_ddp.init_weights()
        # if self.cfg['model']['pretrained']: #and (not self.cfg['model']['resume']):
        #     load_pretrained(self.cfg, self.model_without_ddp, self.logger)
        #     F1score = self.val(0)
        #     self.logger.info(f"F1 of the network on the {self.len_val_loader} test images: {F1score:.1f}%")
        
        lr_scheduler = build_lr_scheduler(self.cfg)(self.optimizer, **self.cfg['solver']['lr_scheduler']['args'])
        val_peek_list = [-1]
        for t in range(self.epoch):
            self.train_loader.sampler.set_epoch(t)
            if torch.distributed.get_rank() == 0:
                self.logger.info(f'==> epoch {t + 1}')
            self.model.train()

            pred_list = list()
            label_list = list()
            mean_loss = 0.0

            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                a = data[0]
                print(a.shape)
                train_label = data[1]
                train1 = []
                train2 = []
                # train_input = np.array(data[0])#.cuda(self.local_rank)
                # train_label = np.array(data[1])#.cuda(self.local_rank)
                # print(train_input.shape)
                # train_input = torch.tensor(train_input).transpose(0,3,1,2)
                # for i in range(len(train_input)):
                #     for j in range(len(train_input[i])):
                #         print(train_input[i][j].shape)
                #         imgs = np.array(train_input[i][j]).transpose(2,0,1)
                #         print(imgs.shape)
                #         train1.append(self.transform(torch.Tensor(imgs)))
                #     train2.append(train2)
                a_flat = a.swapaxes(1,2).flatten(0,1)     # (B*D, C,  H,   W)
                # print(a_flat.shape)
                c = transforms.functional.resize(a_flat , (224, 224))        # (B*D, C, 200, 200)
                # print(c.shape)
                b = b.view(*a.shape[:3], *c.shape[-2:])
                # print(b.shape)
                # train_input = torch.tensor(b).transpose(0,3,1,2)
                train_input = train2.cuda(self.local_rank)
                #.transpose(0,4,1,2,3)
                train_label = torch.from_numpy(np.array(data[1]).astype(np.int64)).cuda(self.local_rank)
                # print(train_input.shape)
                # print(train_input.shape)
                train_output = self.model(train_input)
                # print(train_label.shape)
                train_output = train_output.view(-1, 8)
                train_label = train_label.view(-1)
                # loss = self.criterion(logits1, logits2, label)
                train_loss = self.criterion(train_output, train_label)
                
                mean_loss += train_loss.item()
                train_pred = train_output.argmax(dim=-1)
                # print(len(train_pred))
                if torch.distributed.get_rank() == 0:
                    self.logger.info(f'epoch: {t + 1}/{self.epoch}, iteration: {i + 1}/{self.len_train_loader}, loss: {train_loss.item() :.4f}')
                    
                train_loss.backward()
                self.optimizer.step()

                batch_pred = [torch.zeros_like(train_pred) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_pred, train_pred)
                pred_list.append(torch.cat(batch_pred, dim=0).detach().cpu())

                batch_label = [torch.zeros_like(train_label) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_label, train_label)
                label_list.append(torch.cat(batch_label, dim=0).detach().cpu())
            pred_list = torch.cat(pred_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
            pred_list = pred_list.numpy()
            label_list = label_list.numpy()
            metric_dict = self.metric_fn(**{'pred': pred_list, 'gt': label_list})
            mean_loss = mean_loss / self.len_train_loader

            print_dict = dict()
            print_dict.update({'epoch': f'{t + 1}/{self.epoch}'})
            print_dict.update({'mean_loss': mean_loss})
            print_dict.update({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
            print_dict.update(metric_dict)

            print_str = format_print_dict(print_dict)

            if torch.distributed.get_rank() == 0:
                self.logger.info(f"==> train: {print_str}")
            
                peek = self.val(t + 1)
                if peek > max(val_peek_list):
                    self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, self.task, self.epoch_prefix)
                    val_peek_list.append(peek)
                else:
                    if self.epoch_prefix:
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, self.task, self.epoch_prefix)
                    val_peek_list.append(peek)

            lr_scheduler.step(t)

    
        max_f1 = max(val_peek_list)
        max_f1_epoch = val_peek_list.index(max(val_peek_list)) / 2
        self.logger.info(f'==> End Training, BEST F1: {max_f1}, BEST F1 Epoch: {max_f1_epoch}')

        return max_f1, max_f1_epoch
    
    @torch.no_grad()
    def val(self, t):
        self.model.eval()

        pred_list = list()
        label_list = list()
        save_dict = dict()
        val_mean_loss = 0.0
        save_dict['pred_list'] = list()
        save_dict['label_list'] = list()

        for i, data in enumerate(self.val_loader):
            
            # val_input = data[0]
            a = data[0]
            a_flat = a.flatten(0,1)     # (B*D, C,  H,   W)
            # print(a_flat.shape)
            c = transforms.functional.resize(a_flat , (224, 224))        # (B*D, C, 200, 200)
            # print(c.shape)
            b = c.view(*a.shape[:3], *c.shape[-2:])
            # print(b.shape)
            
            val2 = np.array(b).transpose(0,2,1,3,4)
            val_label = data[1]
            val_input = torch.from_numpy(np.array(val2).astype(np.float32)).cuda(self.local_rank)
            val_label = torch.from_numpy(np.array(val_label).astype(np.int64)).cuda(self.local_rank)
            
            val_output = self.model(val_input)

            val_output = val_output.view(-1, 8)
            val_output2 = torch.stack([val_output,val_output])
            # print(val_output2.shape)
            val_output2 = val_output2.transpose(1,0)
            # print(val_output2.shape)
            val_output = val_output2.reshape(64,8)
            # print(val_output)
            val_label = val_label.view(-1)
            
            val_loss = self.criterion(val_output, val_label)
                
            val_mean_loss += val_loss.item()
            
            val_pred = val_output.argmax(dim=-1)

            pred_list.append(val_pred.detach().cpu())
            label_list.append(val_label.detach().cpu())
            
        
        pred_list = torch.cat(pred_list, dim=0)
        print(pred_list)
        label_list = torch.cat(label_list, dim=0)
        pred_list = pred_list.numpy()
        label_list = label_list.numpy()
        
        metric_dict = self.metric_fn(**{'pred': pred_list, 'gt': label_list})
        print_dict = dict()
        # , loss: {train_loss.item() :.4f}'
        
        print_dict.update({'epoch': f'{t}'})
        print_dict.update({'val_loss': f'{val_loss.item() :.4f}'})
        print_dict.update(metric_dict)
        print_str = format_print_dict(print_dict)

        if torch.distributed.get_rank() == 0:
            self.logger.info(f"==> val: {print_str}")

    
        peek = metric_dict['F1']
        return peek

    def run(self):
        out = self.train()

        return out
    
    @staticmethod
    def save_dict_to_yaml(dict_value, save_path):
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(dict_value, file, sort_keys=False)

    def save_checkpoint(self, model, cfg, log_path, epoch_id, task_name, epoch_prefix=False):
        model.eval()
        if epoch_prefix:
            torch.save(model.module.state_dict(), os.path.join(log_path, f'ckpt_epoch_{epoch_id}_{task_name}.pt'))
        else:
            torch.save(model.module.state_dict(), os.path.join(log_path, f'ckpt_{task_name}.pt'))

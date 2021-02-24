import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib
from ignite.contrib.metrics import ROC_AUC
from loss import consistency_loss
import copy
import numpy as np

def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y

class FixMatch:
    def __init__(self, net, num_classes, ema_m, T, p_cutoff_pos, p_cutoff_neg, lambda_u, labeled_criterion, unlabeled_criterion,\
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """
        
        super(FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        
        self.train_model = net 
        self.eval_model = net
        self.num_eval_iter = num_eval_iter
        self.t_fn = T #temperature params function
        self.p_fn_pos = p_cutoff_pos #confidence cutoff function
        self.p_fn_neg = p_cutoff_neg
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        
        self.optimizer = None
        self.scheduler = None

        self.labeled_criterion = labeled_criterion
        self.unlabeled_criterion = unlabeled_criterion
        
        self.it = 0
        
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        
        # TODO: we don't use ema now so comment this code block
        # for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
        #     param_k.data.copy_(param_q.detach().data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient for eval_net
        #
        # self.eval_model.eval()
            
            
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.module.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)            
    
     
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    
            
    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    
    def train(self, args, logger=None):
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        #lb: labeled, ulb: unlabeled
        # self.train_model.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        for (x_lb, y_lb), (x_ulb_s, x_ulb_w, _) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()
            
            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]
            
            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # self.print_fn(f'{self.it}: concat data done')

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits = self.train_model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                del logits
                # self.print_fn(f'{self.it}: get logits output done')

                # hyper-params for update
                T = self.t_fn
                p_cutoff_pos = self.p_fn_pos
                p_cutoff_neg = self.p_fn_neg

                sup_loss = self.labeled_criterion(args, logits_x_lb, y_lb)
                # self.print_fn(f'{self.it}: get supervised loss done')

                unsup_loss, mask = consistency_loss(args,
                                                logits_x_ulb_w, 
                                                logits_x_ulb_s,
                                                self.unlabeled_criterion,
                                                p_cutoff_pos,
                                                p_cutoff_neg,
                                                'BCE', T, 
                                               use_hard_labels=args.hard_label)

                # self.print_fn(f'{self.it}: get unsupervised loss done')                               

                total_loss = sup_loss + self.lambda_u * unsup_loss
            
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                # self.print_fn(f'{self.it}: use amp and update done')
            else:
                total_loss.backward()
                # self.print_fn(f'{self.it}: backward done')
                self.optimizer.step()
                
            self.scheduler.step()
            self.train_model.zero_grad()
            
            # TODO: we don't use ema so commtent this code block
            # with torch.no_grad():
            #     self._eval_model_update()
                # self.print_fn(f'{self.it}: eval model done')
            
            end_run.record()
            torch.cuda.synchronize()
            # self.print_fn(f'{self.it}: synchronize done')
            
            #tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach() 
            tb_dict['train/unsup_loss'] = unsup_loss.detach() 
            tb_dict['train/total_loss'] = total_loss.detach() 
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach() 
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.
            
            
            if self.it % self.num_eval_iter == 0:
                eval_dict,auc_dict = self.evaluate(args=args)
                # self.print_fn(f'{self.it}: evaluate model done')
                tb_dict.update(eval_dict)
                
                save_path = args.store_path
                
                if tb_dict['eval/auc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/auc']
                    best_it = self.it
                
                self.print_fn(f"{self.it:3} iteration,  SUP_LOSS:{tb_dict['train/sup_loss']:6.4f}, UNSUP_LOSS:{tb_dict['train/unsup_loss']:6.4f}, MASK_RATIO:{tb_dict['train/mask_ratio']}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")
                self.print_fn(f"AUC_list:{auc_dict['eval/auc_list']}")

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                
                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)
                
                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)
                
            self.it +=1
            del tb_dict
            start_batch.record()
            if self.it > 2**19:
                self.num_eval_iter = 1000
        
        eval_dict, _ = self.evaluate(args=args)
        eval_dict.update({'eval/best_auc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict
            
            
    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        # TODO: we don't use ema so commtent this code block
        # use_ema = hasattr(self, 'eval_model')
        # self.print_fn(f'{self.it}: use_ema:{use_ema}')
        # 
        # eval_model = self.eval_model if use_ema else self.train_model
        eval_model = self.eval_model
        eval_model.eval()
        # self.print_fn(f'{self.it}: model evaluate done')
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_num = 0.0
        roc_auc = ROC_AUC(activated_output_transform)
        roc_auc.reset()
        roc_auc_list = []
        for i in range(self.num_classes):
            roc_auc_list.append(ROC_AUC(activated_output_transform))
            roc_auc_list[-1].reset()

        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)
            # self.print_fn(f'{self.it}:get logits for eval_model done')
            
            loss = self.labeled_criterion(args, logits, y)
            # self.print_fn(f'{self.it}:labeled_criterion done')
            
            roc_auc.update((logits.data,y.data))
            for j in range(self.num_classes):
                roc_auc_list[j].update((logits.data[:,j],y.data[:,j]))
            
            total_loss += loss.detach()*num_batch
        
        # TODO: we don't use ema so commtent this code block
        # if not use_ema:
        eval_model.train()
            
        return {'eval/loss': total_loss/total_num, 'eval/auc': roc_auc.compute()},{'eval/auc_list':np.array([x.compute() for x in roc_auc_list])}
    
    
    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")
    
    
    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")

if __name__ == "__main__":
    pass
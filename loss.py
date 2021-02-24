import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
# import ipdb


def get_category_list(annotations, num_classes):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    
    negative_list = dict()
    positive_list = dict()
    cat_list = []

    # initialize dict
    for key in annotations[0].keys():
        positive_list[key] = 0
        negative_list[key] = 0

    for anno in annotations:
        cat = []
        for key in anno.keys():
            # ipdb.set_trace()
            if key not in ['path','Sex','Age','Frontal/Lateral','AP/PA'] and type(anno[key])!=str and anno[key] >= 1:
                positive_list[key] += 1
                cat.append(key)
            
            if key not in ['path','Sex','Age','Frontal/Lateral','AP/PA'] and type(anno[key])!=str and anno[key] == 0:
                negative_list[key] += 1
        cat_list.append(cat)
    
    for i, key in enumerate(negative_list.keys()):
        assert negative_list[key] != 0,f"negative_list[{key}]=0 error"
        num_list[i] = positive_list[key]/negative_list[key]
    return num_list,cat_list


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, num_class_list):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_class_list = num_class_list

    def forward(self, args, output, target, reduction='sum'):
        # output.size = (batchsize, num_classes)
        # pos_weight.size = (num_classes,)
        num_classes = output.shape[1]
        # assert num_classes == 5,f'{output.shape}'
        
        # TODO: num_class_list
        pos_weight = torch.from_numpy(
            np.array(self.num_class_list,
                     dtype=np.float32)).cuda(args.gpu).type_as(output)

        target = target.cuda(args.gpu).type_as(output)
        
        loss_list = []
        for i in range(num_classes):
            target_i = target[:, i].view(-1)
            if reduction == 'expand':
                loss_list.append(F.binary_cross_entropy_with_logits(output[:,i].view(-1), target_i,pos_weight[i],reduction='none'))
            else:
                loss_list.append(F.binary_cross_entropy_with_logits(output[:,i].view(-1), target_i,pos_weight[i]))

        if reduction=='sum':
            loss = torch.sum(torch.stack(loss_list))
            return loss
        elif reduction=='expand':
            return torch.stack(loss_list,dim=1)
        else:
            return loss_list


def consistency_loss(args, logits_w, logits_s, criterion, p_cutoff_pos, p_cutoff_neg, name='BCE', T=1.0, use_hard_labels=True):
    assert name in ['ce', 'L2','BCE']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'BCE':
        # ipdb.set_trace()
        pseudo_label = torch.sigmoid(logits_w)
        # assert pseudo_label.size()[1] == 14,f'{pseudo_label.size()}'
        hard_label = torch.zeros(pseudo_label.size()).cuda(args.gpu)
        mask = torch.zeros(pseudo_label.size()).cuda(args.gpu)

        for i,(p_cutoff_pos_i,p_cutoff_neg_i) in enumerate(zip(p_cutoff_pos,p_cutoff_neg)):
            positive = pseudo_label[:,i].ge(p_cutoff_pos_i).bool()
            negative = pseudo_label[:,i].le(p_cutoff_neg_i).bool()
            mask[:,i] = torch.add(positive.float(),negative.float())
            hard_label[:,i][positive] = 1.0

        if use_hard_labels:
            masked_loss = criterion(args, logits_s, hard_label, reduction='expand') * mask
            return masked_loss.mean(dim=0).sum(), mask.mean()
        else:
            pseudo_label = torch.sigmoid(logits_w/T)
            masked_loss = criterion(args, logits_s, pseudo_label, reduction='expand') * mask
            return masked_loss.mean(dim=0).sum(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')
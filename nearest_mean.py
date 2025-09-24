import argparse
import os
import random
import time

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm
#from torch import linalg 
from torch.nn import functional as F
import torch.nn as nn
import pickle
from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path



class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits=logits.to(device)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask=mask.to(device)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask=logits_mask.to(device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

def TextEmbedLabel(class_labels,textEmbed):
    batchTextEmbed=[]
    for label in class_labels:
        text_feat=textEmbed[label]
        #print("text shape",text_feat.size())
        batchTextEmbed.append(text_feat)
    batchTextEmbed = torch.stack(batchTextEmbed, dim=1).to(device)
    batchTextEmbed=batchTextEmbed.T
    #print("batchTextEmbed shape",batchTextEmbed.size())
    return batchTextEmbed


def ComputeNearestPrototype(class_labels,nearest_mean):
    batchProto=[]
    for label in class_labels:
        mean_feat=nearest_mean[label]
        #print("text shape",text_feat.size())
        batchProto.append(mean_feat)
    batchProto = torch.stack(batchProto, dim=1).to(device)
    batchProto=batchProto.T
    #print("batchTextEmbed shape",batchTextEmbed.size())
    return batchProto

def CosineSim(textEmbed,img_feat):
    sim_logits = img_feat @ textEmbed
    #print("logits shape",logits.size())
    values,idx= torch.topk(sim_logits, dim=-1, largest=True, k=1)
    return values,idx

#regularizer

def orthogonality_regularizer(prototypes):
    """
    Computes the Orthogonality Regularizer loss.

    Args:
        prototypes (torch.Tensor): Tensor of shape (num_classes, feature_dim).

    Returns:
        torch.Tensor: Scalar tensor representing the regularization loss.
    """
    num_classes = prototypes.size(0)
    reg_loss = 0.0

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dot_product = torch.dot(prototypes[i], prototypes[j])
            reg_loss += dot_product ** 2

    return reg_loss


def cosine_similarity_regularizer(prototypes):
    """
    Computes the Cosine Similarity Regularizer loss.

    Args:
        prototypes (torch.Tensor): Tensor of shape (num_classes, feature_dim).

    Returns:
        torch.Tensor: Scalar tensor representing the regularization loss.
    """
    num_classes = prototypes.size(0)
    reg_loss = 0.0

    # Normalize prototypes
    prototypes_norm = prototypes / prototypes.norm(p=2, dim=1, keepdim=True)

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            cosine_similarity = torch.dot(prototypes_norm[i], prototypes_norm[j])
            reg_loss += cosine_similarity ** 2

    return reg_loss


def inverse_distance_regularizer(prototypes, epsilon=1e-8):
    """
    Computes the Inverse Distance Regularizer loss.

    Args:
        prototypes (torch.Tensor): Tensor of shape (num_classes, feature_dim).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        torch.Tensor: Scalar tensor representing the regularization loss.
    """
    num_classes = prototypes.size(0)
    reg_loss = 0.0

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            distance_squared = torch.norm(prototypes[i] - prototypes[j], p=2) ** 2
            reg_loss += 1.0 / (distance_squared + epsilon)

    return reg_loss


def minimum_distance_regularizer(prototypes):
    """
    Computes the Minimum Distance Regularizer loss.

    Args:
        prototypes (torch.Tensor): Tensor of shape (num_classes, feature_dim).

    Returns:
        torch.Tensor: Scalar tensor representing the regularization loss.
    """
    num_classes = prototypes.size(0)
    reg_loss = 0.0
    distances = []

    # Compute all pairwise distances and collect them
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            distance = torch.norm(prototypes[i] - prototypes[j], p=2)
            distances.append(distance)

    # Convert list of distances to tensor
    distances = torch.stack(distances)
    
    # Compute d_mean
    d_mean = torch.mean(distances)

    # Compute the penalty for distances less than d_mean
    penalties = torch.relu(d_mean - distances)
    reg_loss = torch.sum(penalties ** 2)

    return reg_loss


def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args):

    optimizer = SGD(list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0
    textEmbed=pickle.load(open("roberta_cub.p","rb"))
    textEmbed=textEmbed.T
    #print("text Embed: ",textEmbed.size())
    #textEmbed = textEmbed[0:80,:]

    textEmbed,_ = [t for t in textEmbed.chunk(2)]
    textEmbed=textEmbed.to(torch.float32)
    textEmbed=textEmbed.to(device)
    #print("text Embed: ",textEmbed.size())

    nearest_mean=torch.zeros(textEmbed.size())
    margin_text=0
    margin_image=0

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()
        projection_head.train()
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):

            images, class_labels, uq_idxs = batch
            #print("images shape",len(images))
            class_labels = class_labels.to(device)
            images = torch.cat(images, dim=0).to(device)
            img_feat = model(images)
            #print("image_feat size",img_feat.size())
            img_feat=img_feat.to(device)

            img_feat,logits = projection_head(img_feat)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            #print("proj_feat size",img_feat.size())
            #features = torch.nn.functional.normalize(features, dim=-1)

            batchText = TextEmbedLabel(class_labels,textEmbed)
            f1, f2 = [f for f in img_feat.chunk(2)]

            if epoch > 10 :
            
                batchProto = ComputeNearestPrototype(class_labels,nearest_mean)
               # print("batchProto",bat.size())
                #supervised contrastive loss

                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1),batchText.unsqueeze(1),batchProto.unsqueeze(1)], dim=1)
                #sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1),batchText.unsqueeze(1)], dim=1)
                #sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1),batchProto.unsqueeze(1)], dim=1)
                #print("sup_con_feats",sup_con_feats.size())
                #sup_con_labels = class_labels
                #sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
                #sup_con_loss=sup_con_loss.to(device)

            
                #loss = sup_con_loss+margin_image+margin_text
                #loss = sup_con_loss

            
            else:
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1),batchText.unsqueeze(1)], dim=1)
                #sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                #print("sup_con_feats",sup_con_feats.size())
            sup_con_labels = class_labels
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            sup_con_loss=sup_con_loss.to(device)
            
            loss = sup_con_loss


            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))


        with torch.no_grad():

            #Finding the variable threshold
            cos_value=torch.zeros(textEmbed.size(0))
            count_samples=torch.zeros(textEmbed.size(0))

            #Mean of each class and corresponding cosine similarity
            centroid=torch.zeros(textEmbed.size())
            centroid=centroid.to(device)
            #print("mean size",mean.size())
            mean_cos_val=torch.zeros(textEmbed.size(0))
            cosine_sim_txt={}
            cosine_sim_image={}
            for i in range(textEmbed.size(0)):
                cosine_sim_txt[i]=[]
                cosine_sim_image[i]=[]
            #print("cosine_sim_image",cosine_sim_image)
            #print("cosine_sim_txt",cosine_sim_txt)
            for batch_idx, batch in enumerate(train_loader):
                
                images, class_labels, uq_idxs = batch
                #print("images shape",len(images))
                class_labels = class_labels.to(device)
                images = torch.cat(images, dim=0).to(device)
                img_feat = model(images)
                #print("image_feat size",img_feat.size())
                img_feat=img_feat.to(device)

                img_feat,logits = projection_head(img_feat)
                img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
                #print("proj_feat size",img_feat.size())
                
                f1, f2 = [f for f in img_feat.chunk(2)]
                cos_sim= f1 @ textEmbed.T
                f1=f1.to(device)
                
                #print("cos_sim size",cos_sim.size())
               
                for i in range(f1.size(0)):
                    
                    index= class_labels[i]
                    #print("index",index.item())
                    #Mean
                    centroid[index]=centroid[index]+f1[i]                    
                    temp=cos_sim[i,index]
                    #print("value",temp)
                    cosine_sim_txt[index.item()].append(temp.item())
                    cos_value[index]=cos_value[index]+temp
                    count_samples[index]=count_samples[index]+1
                                
            
            cos_value=cos_value/count_samples
            count_samples=count_samples.to(device)
            centroid=centroid.T/count_samples
            centroid = centroid.T
            
            #Mean_cos_sim value
            
            for batch_idx, batch in enumerate(train_loader):
                
                images, class_labels, uq_idxs = batch
                #print("images shape",len(images))
                class_labels = class_labels.to(device)
                images = torch.cat(images, dim=0).to(device)
                img_feat = model(images)
                #print("image_feat size",img_feat.size())
                img_feat=img_feat.to(device)

                img_feat,logits = projection_head(img_feat)
                img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
                #print("proj_feat size",img_feat.size())
                
                f1, f2 = [f for f in img_feat.chunk(2)]
                mcos_sim= f1 @ centroid.T
                
                for i in range(f1.size(0)):
                    
                    index= class_labels[i]
                    #print("index",index)
                    #Mean
                                       
                    temp=mcos_sim[i,index]
                    #print("value",temp)
                    cosine_sim_image[index.item()].append(temp.item())
                    mean_cos_val[index]=mean_cos_val[index]+temp
            mean_cos_val=mean_cos_val.to(device)
            mean_cos_val=mean_cos_val/count_samples
            #print("mean cos thresholds :",mean_cos_val)      
            avg_threshold_txt=torch.zeros(textEmbed.size(0))  
            std_threshold_txt=torch.zeros(textEmbed.size(0)) 
            avg_threshold_image=torch.zeros(textEmbed.size(0))  
            std_threshold_image=torch.zeros(textEmbed.size(0))                    
            #print("cosine_sim_image",cosine_sim_image)
            #print("cosine_sim_txt",cosine_sim_txt)
            for i in range(textEmbed.size(0)):
                temp=torch.tensor(cosine_sim_txt[i])
                #print("temp",temp)
                mean = torch.mean(temp)
                std = torch.std(temp)
                avg_threshold_txt[i]=mean
                std_threshold_txt[i]=std

                temp_image=torch.tensor(cosine_sim_image[i])
                m=torch.mean(temp_image)
                s=torch.std(temp_image)

                avg_threshold_image[i]=m
                std_threshold_image[i]=s

            print("avg_threshold_image",avg_threshold_image)
            print("std_threshold_image",std_threshold_image)
                
            print("mean size",centroid.size())

            #margin_text=orthogonality_regularizer(textEmbed)
            #margin_image=orthogonality_regularizer(centroid)

            margin_text=inverse_distance_regularizer(textEmbed)
            margin_image=inverse_distance_regularizer(centroid)

            #margin_text=cosine_similarity_regularizer(textEmbed)
            #margin_image=cosine_similarity_regularizer(centroid)



            #print("mean",centroid)
            #print("count_samples",count_samples)
            nearest_mean=centroid





            print('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test_on_the_fly(model, projection_head, unlabelled_train_loader,
                                                    epoch=epoch,mean=centroid,avg_threshold=avg_threshold_image,std_threshold=std_threshold_image,k=args.k, save_name='Train ACC Unlabelled',
                                                    args=args)
            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_on_the_fly(model, projection_head, test_loader, epoch=epoch,mean=centroid,avg_threshold=avg_threshold_image,std_threshold=std_threshold_image,k=args.k,save_name='Test ACC', args=args)

        # ----------------

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                              new_acc))
        print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        torch.save(model.state_dict(), args.model_path)
        print("model saved to {}.".format(args.model_path))

        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
        print("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))

        if old_acc_test > best_test_acc_lab:

            print(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
            print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                  new_acc))

            torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
            print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))

            best_test_acc_lab = old_acc_test


def test_on_the_fly(model, projection_head, test_loader,
                epoch,mean,avg_threshold,std_threshold,k,save_name,
                args):

    model.eval()
    projection_head.eval()
    avg_threshold=avg_threshold.to(device)
    std_threshold=std_threshold.to(device)
    print("k=",k)
    variable_threshold=avg_threshold-k*std_threshold
    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.to(device)
        #print(label)

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        #_, feats, _ = projection_head(feats)
        feats,logits = projection_head(feats)

        feats = F.normalize(feats, dim=-1)[:, :]
#         print(feats.shape)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # On-The-Fly
    # -----------------------
    #textEmbed=pickle.load(open("sbertcifar100.p","rb"))
    textEmbed=mean
    #textEmbed=textEmbed.T
    print("text Embed: ",textEmbed.size())
    #textEmbed = textEmbed[0:80,:]

    #textEmbed,_ = [t for t in textEmbed.chunk(2)]
    #textEmbed=textEmbed.to(torch.float32)
    textEmbed=textEmbed.to(device)
    class_seen=textEmbed.size(0)
    #print(class_seen)
    
    
    all_feats = np.concatenate(all_feats)
    #print("all_feats",all_feats.shape)
    feat_list=all_feats.tolist()
    #print("target: ", targets[0:2])
    preds = []
    count=0
    variable_threshold=avg_threshold-k*std_threshold
    for feat in feat_list:
        count=count+1
        #print(feat)
        feat=torch.Tensor(feat).to(device)
        #print("feat shape ",feat.size())
        
        cos_sim= feat @ textEmbed.T
        #print("cos_sim size",cos_sim.size())
        variable_threshold=avg_threshold-k*std_threshold
        diff_text=cos_sim - variable_threshold
        #diff_image=mean_cos_sim -  variable_threshold_image
        #print("diff:",diff)
        thres_val_text,index_text =  torch.topk(diff_text, dim=-1, largest=True, k=1)
        #thres_val_image,index_image =  torch.topk(diff_image, dim=-1, largest=True, k=1)
        if thres_val_text>=0:
            
                preds.append(index_text)
        else:
        
                feat=feat.unsqueeze(0)
                #print(feat.size())
                #print(thres_val)
                textEmbed=torch.cat([textEmbed,feat],dim=0)
                
                new_threshold_avg=avg_threshold[index_text]
                new_threshold_std=std_threshold[index_text]

                #new_threshold_image=variable_threshold_image[index_image]
                #print("new_threshold:",new_threshold)
                avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                std_threshold=torch.cat([std_threshold,new_threshold_std])
                ##variable_threshold_image=torch.cat([ variable_threshold_image,new_threshold_image])
                
                preds.append(textEmbed.size(0))
            
    variable_threshold=avg_threshold-k*std_threshold
    print("variable threshold for old classes:",variable_threshold[:class_seen])
    print("variable threshold for novel classes",variable_threshold[class_seen:])
    print("textEmbed shape",textEmbed.size())

    #print("image variable threshold for old classes:",variable_threshold_image[:class_seen])
    #print("image variable threshold for novel classes",variable_threshold_image[class_seen:])
    #print("Mean shape",mean.size())

    preds=torch.Tensor(preds)
    #print("preds shape",preds.size())
    preds = preds.cpu().numpy()
    print(len(list(set(preds))), len(preds))

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--k', default=2, type=float)
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    
        # ----------------------
        # Multiple Runs
        # ----------------------
    for run in range(0, 1):

        # ----------------------
        # INIT
        # ----------------------
        seed_torch(run)
        args = parser.parse_args()
        device = torch.device('cuda:2')
        args = get_class_splits(args)
        print("Introducing prototypes as view after 10 epochs, mean-k*std,k=4")

        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(args.unlabeled_classes)

        init_experiment(args, runner_name=['checkpoints'])
        print(f'Using evaluation function {args.eval_funcs[0]} to print results')

        # ----------------------
        # BASE MODEL
        # ----------------------
        if args.base_model == 'vit_dino':

            args.interpolation = 3
            args.crop_pct = 0.875
            pretrain_path = dino_pretrain_path

            model = vits.__dict__['vit_base']()
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

            #state_dict = torch.load(pretrain_path, map_location='cpu')
            #model.load_state_dict(state_dict)

            if args.warmup_model_dir is not None:
                print(f'Loading weights from {args.warmup_model_dir}')
                model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

            model.to(device)

            # NOTE: Hardcoded image size as we do not finetune the entire ViT model
            args.image_size = 224
            args.feat_dim = 768
            args.num_mlp_layers = 3
            args.code_dim = 12
            args.mlp_out_dim = 768


            # ----------------------
            # HOW MUCH OF BASE MODEL TO FINETUNE
            # ----------------------
            for m in model.parameters():
                m.requires_grad = False

            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in model.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True

        else:

            raise NotImplementedError

        # --------------------
        # CONTRASTIVE TRANSFORM
        # --------------------
        train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
        train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

        # --------------------
        # DATASETS
        # --------------------
        train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_dataset = get_datasets(args.dataset_name,
                                                                                             train_transform,
                                                                                             test_transform,
                                                                                             args)



        # --------------------
        # DATALOADERS
        # --------------------
        labelled_train_loader = DataLoader(labelled_dataset, num_workers=args.num_workers, batch_size=args.batch_size, 
                                  shuffle=True, drop_last=True)
        unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                          batch_size=args.batch_size, shuffle=False)

        # ----------------------
        # PROJECTION HEAD
        # ----------------------
        projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                   out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
        projection_head.to(device)

        # ----------------------
        # TRAIN
        # ----------------------
        train(projection_head, model, labelled_train_loader, test_loader, unlabelled_train_loader, args)

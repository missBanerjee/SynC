import argparse
import os
import random
from tempfile import tempdir
import time
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits,get_datasets2
from tqdm import tqdm
import pickle
from torch.nn import functional as F
import torch.nn as nn

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
    
class prototypeClassifier(nn.Module):
  def __init__(self,prototype):
    super().__init__()
    prototype=prototype.to(torch.float32)
    self.protos = nn.Parameter(data=prototype, requires_grad=True)
    

  def forward(self,features): 
    #cosine simsilarity
    
    #print(features.shape)
    features=features.to(torch.float32)
    
    score= torch.matmul(features,torch.t(self.protos))
    
    #normalize the scores
    score= torch.nn.functional.normalize(score, dim=-1)
    scores=20*score
    class_pred= torch.argmax(scores,dim=1)
    #print(score.shape)
    #print(class_pred)
    return score,class_pred

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

def CosineSim(textEmbed,img_feat):
    sim_logits = img_feat @ textEmbed
    #print("logits shape",logits.size())
    values,idx= torch.topk(sim_logits, dim=-1, largest=True, k=1)
    return values,idx

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
    
def compute_mean_var(model,dino_head, train_loader):

    all_feats = []
    targets = np.array([])
    for batch_idx, batch in enumerate(tqdm(train_loader)):

        images, class_labels, uq_idxs = batch
        images = torch.cat(images, dim=0).to(device)
        
        #print(label)
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        feats,logits = dino_head(feats)
        feats = F.normalize(feats, dim=-1)[:, :]
        #print(feats.shape)
        f1, f2 = [f for f in feats.chunk(2)]
        f1=f1.cpu().detach().numpy()
        
        all_feats.append(f1)
        targets = np.append(targets, class_labels.cpu().numpy())
    all_feats = np.concatenate(all_feats)
    #print(all_feats.shape)
   
    targets=torch.from_numpy(targets)
    #print(targets.size())
    all_feats=torch.from_numpy(all_feats)
    #print(all_feats.size())
    labels,indices=torch.sort(targets)
    #print(labels[50])
    #print(indices)
    #all_feats = torch.nn.functional.normalize(all_feats, dim=-1)


    change_indices=np.array([])
    for i in range(labels.size(0)):
        if labels[i-1]!=labels[i]:
            change_indices=np.append(change_indices,i)
      
    change_indices=torch.from_numpy(change_indices)
    #print(change_indices)
    #print(change_indices.size(0))
    #print(labels.size(0))
    features=all_feats[indices]
    
    #COMPUTE CLASS_MEANS
    class_means=[]
    for i in range(change_indices.size(0)):
        
        if(i==(change_indices.size(0)-1)):
            indx1=change_indices[i].to(torch.int)
            indx2=labels.size(0)
            n_count=indx2-indx1
            #print(n_count)
            temp=features[indx1:indx2,:]
            #print(temp.size())
            temp=torch.sum(temp, dim=0)
            #print(temp)
            mean=temp/n_count
            #print(mean)
        
        elif(i!=(change_indices.size(0)-1)):
            indx1=change_indices[i].to(torch.int)
            indx2=change_indices[i+1].to(torch.int)
            n_count=indx2-indx1
            #print(n_count)
            temp=features[indx1:indx2,:]
            #print(temp.size())
            temp=torch.sum(temp, dim=0)
            #print(temp)
            mean=temp/n_count
            #print(mean)
        class_means.append(mean)
        
    class_means=torch.stack(class_means, dim=0)
    #print(class_means.size())
    #COMPUTE CLASS_VARIANCE
    count=0
    class_cov=[]
    for i in range(change_indices.size(0)):

        if(i==(change_indices.size(0)-1)):
            indx1=change_indices[i].to(torch.int)
            indx2=labels.size(0)
            n_count=indx2-indx1
           
            temp=features[indx1:indx2,:]
            mean=class_means[count]
            
           
            cov = torch.cov(torch.tensor(temp, dtype=torch.float64).T)+torch.eye(mean.shape[-1])*1e-5
            #print(temp)
            #print(cov.size())
            count=count+1
            
           
        
        elif(i!=(change_indices.size(0)-1)):
            indx1=change_indices[i].to(torch.int)
            indx2=change_indices[i+1].to(torch.int)
            n_count=indx2-indx1
            
            temp=features[indx1:indx2,:]
            mean=class_means[count]
            

            cov = torch.cov(torch.tensor(temp, dtype=torch.float64).T)+torch.eye(mean.shape[-1])*1e-5
            #print(cov.size())
            count=count+1
            
        class_cov.append(cov)
        
    class_cov=torch.stack(class_cov, dim=0)
    #print(class_cov.size())
    labels= torch.unique(labels)
    #print(labels)
    return class_means,class_cov,labels
def computeMean(model,dino_head,textEmbed,train_loader):
    #Finding the variable threshold
    
    count_samples=torch.zeros(textEmbed.size(0))

    #Mean of each class and corresponding cosine similarity
    centroid=torch.zeros(textEmbed.size())
    centroid=centroid.to(device)
    print("mean size",centroid.size())
    
    model.eval()
    dino_head.eval()

    with torch.no_grad():
    
        for batch_idx, batch in enumerate(train_loader):
                
            images, class_labels, uq_idxs = batch
            #print("images shape",len(images))
            class_labels = class_labels.to(device)
            images = torch.cat(images, dim=0).to(device)
            img_feat = model(images)
            #print("image_feat size",img_feat.size())
            img_feat=img_feat.to(device)

            img_feat,logits = dino_head(img_feat)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            #print("proj_feat size",img_feat.size())
                
            f1, f2 = [f for f in img_feat.chunk(2)]
       
            f1=f1.to(device)
                
            #print("cos_sim size",cos_sim.size())
               
            for i in range(f1.size(0)):
                    
                index= class_labels[i]
                #print("index",index.item())
                #Mean
                centroid[index]=centroid[index]+f1[i]                              
                count_samples[index]=count_samples[index]+1
                                
        count_samples=count_samples.to(device)
        centroid=centroid.T/count_samples
        centroid = centroid.T
        
    return centroid
            
def realign2(cls_means,cls_cov,labels,textEmbed,r_loss,args):
    
    cls_means=torch.Tensor(cls_means)
    cls_cov=torch.Tensor(cls_cov)
    labels=torch.Tensor(labels)
    textEmbed=torch.tensor(textEmbed,requires_grad=True)
    neo_projection_head = prototypeClassifier(textEmbed)
    neo_projection_head.to(device)
    epochs=5
    optimizer = SGD(list(neo_projection_head.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=args.lr * 1e-3,
        )
  
    best_test_acc_lab = 0
    torch.set_grad_enabled(True)

    for epoch in range(epochs):

        

        neo_projection_head.train()
        losses = 0.

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = 128

        for c_id in range(len(labels)):      

            mean=cls_means[c_id].to(device)
            cov=cls_cov[c_id].to(device)
            #multivariate normal dist
            m = MultivariateNormal(mean.float(), cov.float())
            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
            #print(sampled_data_single.size())
            
            sampled_data.append(sampled_data_single)                
            sampled_label.extend([c_id]*num_sampled_pcls)

        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        #print("sampled_data size",sampled_data.size())
        sampled_label = torch.tensor(sampled_label).long().to(device)
        #print("sampled_label size",sampled_label.size())
        #inputs=torch.tensor(sampled_data,requires_grad=True)
        inputs=sampled_data.clone().detach()
        #inputs=sampled_data.clone().detach().requires_grad_(True)
        #inputs = sampled_data
        targets= sampled_label
        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        batch_size=128
        for _iter in range(len(labels)):
            inp = inputs[_iter*batch_size:(_iter+1)*batch_size]
            tgt = targets[_iter*batch_size:(_iter+1)*batch_size]
            #print("tgt",tgt)
            logits,pred=neo_projection_head(inp)

            #print("logits ",logits)
            #print("pred",pred)
            r_loss=r_loss.detach()
            loss = F.cross_entropy(logits, tgt)+r_loss
            #print("logits shape",logits.size())
            #print("pred shape",pred.size())
            #print("loss:",loss)
            #loss.requires_grad = True
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

        #print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,train_acc_record.avg))
         # Step schedule
        exp_lr_scheduler.step()
        neo_protos=neo_projection_head.protos     
    return neo_protos
def CosineSimMargin(textEmbed,img_feat):
    sim_logits = img_feat @ textEmbed
    #print("logits shape",logits.size())
    values,idx= torch.topk(sim_logits, dim=-1, largest=True, k=2)
    return values,idx

def realign(cls_means,cls_cov,labels,textEmbed,args):
    
    cls_means=torch.Tensor(cls_means)
    cls_cov=torch.Tensor(cls_cov)
    labels=torch.Tensor(labels)
    textEmbed=torch.tensor(textEmbed,requires_grad=True)
    neo_projection_head = prototypeClassifier(textEmbed)
    neo_projection_head.to(device)
    epochs=5
    optimizer = SGD(list(neo_projection_head.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=args.lr * 1e-3,
        )
  
    best_test_acc_lab = 0
    torch.set_grad_enabled(True)

    for epoch in range(epochs):

        

        neo_projection_head.train()
        losses = 0.

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = 128

        for c_id in range(len(labels)):      

            mean=cls_means[c_id].to(device)
            cov=cls_cov[c_id].to(device)
            #multivariate normal dist
            m = MultivariateNormal(mean.float(), cov.float())
            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
            #print(sampled_data_single.size())
            
            sampled_data.append(sampled_data_single)                
            sampled_label.extend([c_id]*num_sampled_pcls)

        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        #print("sampled_data size",sampled_data.size())
        sampled_label = torch.tensor(sampled_label).long().to(device)
        #print("sampled_label size",sampled_label.size())
        #inputs=torch.tensor(sampled_data,requires_grad=True)
        inputs=sampled_data.clone().detach()
        #inputs=sampled_data.clone().detach().requires_grad_(True)
        #inputs = sampled_data
        targets= sampled_label
        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        batch_size=128
        for _iter in range(len(labels)):
            inp = inputs[_iter*batch_size:(_iter+1)*batch_size]
            tgt = targets[_iter*batch_size:(_iter+1)*batch_size]
            #print("tgt",tgt)
            logits,pred=neo_projection_head(inp)

            #print("logits ",logits)
            #print("pred",pred)
            
            loss = F.cross_entropy(logits, tgt)
            #print("logits shape",logits.size())
            #print("pred shape",pred.size())
            #print("loss:",loss)
            #loss.requires_grad = True
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

        #print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,train_acc_record.avg))
         # Step schedule
        exp_lr_scheduler.step()
        neo_protos=neo_projection_head.protos     
    return neo_protos

def computeConfusingThreshold(variable_threshold,confusing_label,confusing_buffer):
    con_len=len(confusing_buffer)
    confusing_threshold=[]
    #print("variable_threshold length",variable_threshold.size())
    for i in range(con_len):
        id=confusing_label[i]
        id=torch.tensor(id).item()
        #print("id",id)
        temp=variable_threshold[id]
        #print("")
        confusing_threshold.append(temp)
    confusing_threshold=torch.stack(confusing_threshold,dim=0)
    return confusing_threshold




def train(model,dino_head,projection_head,cls_means,cls_cov,labels,train_loader, test_loader, unlabelled_train_loader, textEmbed,args):

    optimizer = SGD(list(projection_head.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
  
    best_test_acc_lab = 0
    with torch.no_grad():
        model.eval()
        dino_head.eval()
        textEmbed=projection_head.protos
        #Finding the variable threshold
        cos_value=torch.zeros(textEmbed.size(0))
        count_samples=torch.zeros(textEmbed.size(0))

        #Mean of each class and corresponding cosine similarity
        mean=torch.zeros(textEmbed.size())
        mean=mean.to(device)
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

            img_feat,logits = dino_head(img_feat)
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
                mean[index]=mean[index]+f1[i]                    
                temp=cos_sim[i,index]
                #print("value",temp)
                cosine_sim_txt[index.item()].append(temp.item())
                cos_value[index]=cos_value[index]+temp
                count_samples[index]=count_samples[index]+1
                                
            
        cos_value=cos_value/count_samples
        count_samples=count_samples.to(device)
        mean=mean.T/count_samples
        mean = mean.T
        #print("mean size",mean.size())
        #print("mean",mean)
        #Mean_cos_sim value
            
        for batch_idx, batch in enumerate(train_loader):
                
            images, class_labels, uq_idxs = batch
            #print("images shape",len(images))
            class_labels = class_labels.to(device)
            images = torch.cat(images, dim=0).to(device)
            img_feat = model(images)
            #print("image_feat size",img_feat.size())
            img_feat=img_feat.to(device)

            img_feat,logits = dino_head(img_feat)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            #print("proj_feat size",img_feat.size())
                
            f1, f2 = [f for f in img_feat.chunk(2)]
            mcos_sim= f1 @ mean.T
               
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
        #print("cosine_sim_image",cosine_sim_image)
        #print("cosine_sim_txt",cosine_sim_txt)
        for i in range(textEmbed.size(0)):
            temp=torch.tensor(cosine_sim_txt[i])
            mean = torch.mean(temp)
            std = torch.std(temp)
            avg_threshold_txt[i]=mean
            std_threshold_txt[i]=std

        print("avg_threshold_txt",avg_threshold_txt)
        print("std_threshold_txt",std_threshold_txt)
                

    
        print('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test_on_the_fly( model,dino_head,projection_head, unlabelled_train_loader,
                                                    epoch=0,avg_threshold=avg_threshold_txt,std_threshold=std_threshold_txt,k=args.k, save_name='Train ACC Unlabelled',
                                                    args=args)
        print('Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test = test_on_the_fly(model,dino_head,projection_head, test_loader, epoch=0,avg_threshold=avg_threshold_txt,std_threshold=std_threshold_txt,k=args.k, save_name='Test ACC', args=args)

        # ----------------


    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        projection_head.train()
        losses = 0.

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = 128

        for c_id in range(len(labels)):      

            mean=cls_means[c_id].to(device)
            cov=cls_cov[c_id].to(device)
            #multivariate normal dist
            m = MultivariateNormal(mean.float(), cov.float())
            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
            #print(sampled_data_single.size())
            
            sampled_data.append(sampled_data_single)                
            sampled_label.extend([c_id]*num_sampled_pcls)

        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        #print("sampled_data size",sampled_data.size())
        sampled_label = torch.tensor(sampled_label).long().to(device)
        #print("sampled_label size",sampled_label.size())
        
        inputs = sampled_data
        targets= sampled_label
        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        batch_size=128
        for _iter in range(len(labels)):
            inp = inputs[_iter*batch_size:(_iter+1)*batch_size]
            tgt = targets[_iter*batch_size:(_iter+1)*batch_size]
            #print("tgt",tgt)
            logits,pred=projection_head(inp)

            #print("logits ",logits)
            #print("pred",pred)
            
            loss = F.cross_entropy(logits, tgt)
            #print("logits shape",logits.size())
            #print("pred shape",pred.size())

            #loss = F.cross_entropy(logits[:, :], )

            #loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            loss_record.update(loss.item(), tgt.size(0))
        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))

        
        with torch.no_grad():
            textEmbed=projection_head.protos
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

                img_feat,logits = dino_head(img_feat)
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
            #print("mean size",mean.size())
            #print("mean",mean)
            #Mean_cos_sim value
            
            for batch_idx, batch in enumerate(train_loader):
                
                images, class_labels, uq_idxs = batch
                #print("images shape",len(images))
                class_labels = class_labels.to(device)
                images = torch.cat(images, dim=0).to(device)
                img_feat = model(images)
                #print("image_feat size",img_feat.size())
                img_feat=img_feat.to(device)

                img_feat,logits = dino_head(img_feat)
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

            #print("avg_threshold_txt",avg_threshold_txt)
            #print("std_threshold_txt",std_threshold_txt)
                

            #print("min_threshold_txt",min_threshold_txt)
            #print("min_threshold_image",min_threshold_image)
                

            #print("count_samples",count_samples)

        with torch.no_grad():
            print('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test_on_the_fly( model,dino_head,projection_head, unlabelled_train_loader,
                                                    epoch=epoch,avg_threshold=avg_threshold_image,std_threshold=std_threshold_image,k=args.k, save_name='Train ACC Unlabelled',
                                                    args=args)
            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_on_the_fly(model,dino_head,projection_head, test_loader, epoch=epoch,avg_threshold=avg_threshold_image,std_threshold=std_threshold_image,k=args.k, save_name='Test ACC', args=args)

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

    print('Testing on unlabelled examples in the training data...')
    all_acc, old_acc, new_acc = test_on_the_fly_margin_baseline( model,dino_head,projection_head,cls_means,cls_cov,labels, unlabelled_train_loader,avg_threshold=avg_threshold_txt,std_threshold=std_threshold_txt,k=args.k,save_name='Train ACC Unlabelled',
                                                args=args)

    #all_acc, old_acc, new_acc = test_on_the_fly_active8( model,dino_head,projection_head,cls_means,cls_cov,labels, unlabelled_train_loader,avg_threshold=avg_threshold_image,std_threshold=std_threshold_image,k=args.k,save_name='Train ACC Unlabelled',
    #                                                args=args)
    print('Train Accuracies after classifier alignment: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,new_acc))


def test_on_the_fly_active6(model,dino_head,projection_head, cls_means,cls_cov,labels,test_loader
                        ,avg_threshold,std_threshold,k,save_name,args):

    textEmbed=projection_head.protos
    #print("textEmbed",textEmbed.size())
    model.eval()
    dino_head.eval()
    print("k:",k)
    #variable_threshold_txt=variable_threshold_txt.to(device)
    avg_threshold=avg_threshold.to(device)
    std_threshold=std_threshold.to(device)
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
        feats,logits = dino_head(feats)
        feats = F.normalize(feats, dim=-1)[:, :]
        #print(feats.shape)
        feats=feats.cpu().detach().numpy()
        all_feats.append(feats)
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # On-The-Fly
    # -----------------------
    
    cls_cov=cls_cov.tolist()
    cls_means=cls_means.tolist()
    labels=labels.tolist()
    print("len of cls_cov before testing",len(cls_cov))
    print("len of cls_means before testing",len(cls_means))
    print("len of labels before testing",len(labels))


    textEmbed=textEmbed.to(torch.float32)
    textEmbed=textEmbed.to(device)
    class_seen=textEmbed.size(0)
    #print(class_seen)
    
    all_feats = np.concatenate(all_feats)
    #print("all_feats",all_feats.shape)
    feat_list=all_feats.tolist()
    #print("target: ", targets[0:2])
    preds = []
    confusing_label={}
    textEmbedNovel={}
    for i in range(3*textEmbed.size(0)):
        textEmbedNovel[i]=[]
        confusing_label[i]=[]
    confusing_buffer=[]
    count_novel=0
    count=0
    old=0
    seen=0
    bud=0
    out_of_bud=0
    budget=2.5*class_seen
    count_sample=-1
    unseen=0
    flag=0
    for feat in feat_list:
        #print(feat)
        feat=torch.Tensor(feat).to(device)
        #print("feat shape ",feat.size())
        count_sample=count_sample+1
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
            if len(confusing_buffer)>=2:
                #print("Number of confusing samples collected:",len(confusing_samples))
                confusing_samples= torch.stack(confusing_buffer,dim=0)
                confusion_cos_sim=feat @ confusing_samples.T
                #print("confusion cos sim",confusion_cos_sim)
                confusing_threshold=computeConfusingThreshold(variable_threshold,confusing_label,confusing_buffer)
                #print("confusing threshold",confusing_threshold)
                con_diff_text=confusion_cos_sim - confusing_threshold
                con_val,con_index =  torch.topk(con_diff_text, dim=-1, largest=True, k=1)
                if con_val>=0:
                    temp=confusing_label[con_index.item()]
                    temp=torch.tensor(temp).item()
                    #print("temp",temp)
                    preds.append(temp)
                else:
                    if(bud<budget):
                        bud=bud+1
                        feat_label=targets[count_sample]
                        feat_label= int(feat_label)

                        
                        if feat_label<class_seen:
                            preds.append(feat_label)
                            old=old+1
                            confusing_buffer.append(feat)
                            con_len=len(confusing_buffer)-1
                            confusing_label[con_len].append(feat_label)
                            #flag=1
                        elif feat_label>=class_seen:
                            countNovel=textEmbed.size(0)-class_seen
                            
                            #print("countNovel",countNovel)
                            if countNovel!=0:
                                flag=0
                                for i in range(countNovel):
                                    temp=torch.tensor(textEmbedNovel[i]).item()
                                    #print("novel label",temp)
                                    if(feat_label==temp):
                                        preds.append(i+class_seen)
                                        seen=seen+1
                                        confusing_buffer.append(feat)
                                        con_len=len(confusing_buffer)-1
                                        confusing_label[con_len].append(i+class_seen)
                                        flag=1
                                        #print("hi before break")
                                        break
                                    #print("hi from if loop")
                                #print("hi from for loop")

                                if(flag!=1):
                                    #print("hi from flag")
                                    unseen=unseen+1
                                    feat=feat.unsqueeze(0)
                                    feat_label=targets[count_sample]
                                    feat_label= int(feat_label)
                                    #print(feat.size())
                                    #print(thres_val)
                                    textEmbed=torch.cat([textEmbed,feat],dim=0)
                
                                    new_threshold_avg=avg_threshold[index_text]
                                    new_threshold_std=std_threshold[index_text]

                            
                                    #print("new_threshold:",new_threshold)
                                    avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                                    std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                                    neo_label=textEmbed.size(0)
                                    preds.append(neo_label)
                                    #print("feat_label",feat_label)
                                    novel=neo_label-class_seen-1
                                    #print("novel",novel)
                                    textEmbedNovel[novel].append(feat_label)
                                    #print(textEmbedNovel)
                                    id=random.randint(0, class_seen-1)
                                    cov=cls_cov[id]
                                    cls_cov.append(cov)
                                    labels.append(neo_label)


                                    count_novel=count_novel+1

                                    if(count_novel-count)==10:
                                        for i in range(class_seen+count,class_seen+count_novel):
                                            cls_means.append(textEmbed[i])
                                        #print("len of cls_means after update",len(cls_means))
                                        #print("len of cls_cov after update",len(cls_cov))
                                        #print("len of labels after update",len(labels))
                                        print("count",count)
                                        print("count novel", count_novel)
                                        count=count_novel
                                        neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                                        textEmbed=neo_protos

                            elif countNovel==0:
                                unseen=unseen+1
                                feat=feat.unsqueeze(0)
                                #print(feat.size())
                                #print(thres_val)
                                feat_label=targets[count_sample]
                                feat_label= int(feat_label)
                                textEmbed=torch.cat([textEmbed,feat],dim=0)

                                new_threshold_avg=avg_threshold[index_text]
                                new_threshold_std=std_threshold[index_text]

        
                                #print("new_threshold:",new_threshold)
                                avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                                std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                                neo_label=textEmbed.size(0)
                                preds.append(neo_label)
                                #print("feat_label",feat_label)
                                novel=neo_label-class_seen-1
                                #print("novel",novel)
                                textEmbedNovel[novel].append(feat_label)
                                #print(textEmbedNovel)
                                id=random.randint(0, class_seen-1)
                                cov=cls_cov[id]
                                cls_cov.append(cov)
                                labels.append(neo_label)


                                count_novel=count_novel+1

                                if(count_novel-count)==10:
                                    for i in range(class_seen+count,class_seen+count_novel):
                                        cls_means.append(textEmbed[i])
                                    #print("len of cls_means after update",len(cls_means))
                                    #print("len of cls_cov after update",len(cls_cov))
                                    #print("len of labels after update",len(labels))
                                    print("count",count)
                                    print("count novel", count_novel)
                                    count=count_novel
                                    neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                                    textEmbed=neo_protos

                    elif bud >= budget :
                                unseen=unseen+1
                                out_of_bud=out_of_bud+1
                                feat=feat.unsqueeze(0)
                                feat_label=targets[count_sample]
                                feat_label= int(feat_label)
                                #print(feat.size())
                                #print(thres_val)
                                textEmbed=torch.cat([textEmbed,feat],dim=0)

                                new_threshold_avg=avg_threshold[index_text]
                                new_threshold_std=std_threshold[index_text]

        
                                #print("new_threshold:",new_threshold)
                                avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                                std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                                neo_label=textEmbed.size(0)
                                preds.append(neo_label)
                                #print("feat_label",feat_label)
                                novel=neo_label-class_seen-1
                                #print("novel",novel)
                                textEmbedNovel[novel].append(feat_label)
                                #print(textEmbedNovel)
                                id=random.randint(0, class_seen-1)
                                cov=cls_cov[id]
                                cls_cov.append(cov)
                                labels.append(neo_label)

                
                                count_novel=count_novel+1

                                if(count_novel-count)==10:
                                    for i in range(class_seen+count,class_seen+count_novel):
                                        cls_means.append(textEmbed[i])
                                    #print("len of cls_means after update",len(cls_means))
                                    #print("len of cls_cov after update",len(cls_cov))
                                    #print("len of labels after update",len(labels))
                                    print("count",count)
                                    print("count novel", count_novel)
                                    count=count_novel
                                    neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                                    textEmbed=neo_protos
                            
            elif len(confusing_buffer)<2:
                    if(bud<budget):
                        bud=bud+1
                        feat_label=targets[count_sample]
                        feat_label= int(feat_label)

                        
                        if feat_label<class_seen:
                            preds.append(feat_label)
                            old=old+1
                            confusing_buffer.append(feat)
                            con_len=len(confusing_buffer)-1
                            confusing_label[con_len].append(feat_label)
                            #flag=1
                        elif feat_label>=class_seen:
                            countNovel=textEmbed.size(0)-class_seen
                            
                            #print("countNovel",countNovel)
                            if countNovel!=0:
                                flag=0
                                for i in range(countNovel):
                                    temp=torch.tensor(textEmbedNovel[i]).item()
                                    #print("novel label",temp)
                                    if(feat_label==temp):
                                        preds.append(i+class_seen)
                                        seen=seen+1
                                        confusing_buffer.append(feat)
                                        con_len=len(confusing_buffer)-1
                                        confusing_label[con_len].append(i+class_seen)
                                        flag=1
                                        #print("hi before break")
                                        break
                                    #print("hi from if loop")
                                #print("hi from for loop")

                                if(flag!=1):
                                    #print("hi from flag")
                                    unseen=unseen+1
                                    feat=feat.unsqueeze(0)
                                    feat_label=targets[count_sample]
                                    feat_label= int(feat_label)
                                    #print(feat.size())
                                    #print(thres_val)
                                    textEmbed=torch.cat([textEmbed,feat],dim=0)
                
                                    new_threshold_avg=avg_threshold[index_text]
                                    new_threshold_std=std_threshold[index_text]

                            
                                    #print("new_threshold:",new_threshold)
                                    avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                                    std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                                    neo_label=textEmbed.size(0)
                                    preds.append(neo_label)
                                    #print("feat_label",feat_label)
                                    novel=neo_label-class_seen-1
                                    #print("novel",novel)
                                    textEmbedNovel[novel].append(feat_label)
                                    #print(textEmbedNovel)
                                    id=random.randint(0, class_seen-1)
                                    cov=cls_cov[id]
                                    cls_cov.append(cov)
                                    labels.append(neo_label)


                                    count_novel=count_novel+1

                                    if(count_novel-count)==10:
                                        for i in range(class_seen+count,class_seen+count_novel):
                                            cls_means.append(textEmbed[i])
                                        #print("len of cls_means after update",len(cls_means))
                                        #print("len of cls_cov after update",len(cls_cov))
                                        #print("len of labels after update",len(labels))
                                        print("count",count)
                                        print("count novel", count_novel)
                                        count=count_novel
                                        neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                                        textEmbed=neo_protos

                            elif countNovel==0:
                                unseen=unseen+1
                                feat=feat.unsqueeze(0)
                                feat_label=targets[count_sample]
                                feat_label= int(feat_label)
                                #print(feat.size())
                                #print(thres_val)
                                textEmbed=torch.cat([textEmbed,feat],dim=0)

                                new_threshold_avg=avg_threshold[index_text]
                                new_threshold_std=std_threshold[index_text]

        
                                #print("new_threshold:",new_threshold)
                                avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                                std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                                neo_label=textEmbed.size(0)
                                preds.append(neo_label)
                                #print("feat_label",feat_label)
                                novel=neo_label-class_seen-1
                                #print("novel",novel)
                                textEmbedNovel[novel].append(feat_label)
                                #print(textEmbedNovel)
                                id=random.randint(0, class_seen-1)
                                cov=cls_cov[id]
                                cls_cov.append(cov)
                                labels.append(neo_label)

                
                                count_novel=count_novel+1

                                if(count_novel-count)==10:
                                    for i in range(class_seen+count,class_seen+count_novel):
                                        cls_means.append(textEmbed[i])
                                    #print("len of cls_means after update",len(cls_means))
                                    #print("len of cls_cov after update",len(cls_cov))
                                    #print("len of labels after update",len(labels))
                                    print("count",count)
                                    print("count novel", count_novel)
                                    count=count_novel
                                    neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                                    textEmbed=neo_protos
                    elif bud>=budget:
                                feat_label=targets[count_sample]
                                feat_label= int(feat_label)
                                unseen=unseen+1
                                out_of_bud=out_of_bud+1
                                feat=feat.unsqueeze(0)
                                #print(feat.size())
                                #print(thres_val)
                                textEmbed=torch.cat([textEmbed,feat],dim=0)

                                new_threshold_avg=avg_threshold[index_text]
                                new_threshold_std=std_threshold[index_text]

        
                                #print("new_threshold:",new_threshold)
                                avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                                std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                                neo_label=textEmbed.size(0)
                                preds.append(neo_label)
                                #print("feat_label",feat_label)
                                novel=neo_label-class_seen-1
                                #print("novel",novel)
                                textEmbedNovel[novel].append(feat_label)
                                #print(textEmbedNovel)
                                id=random.randint(0, class_seen-1)
                                cov=cls_cov[id]
                                cls_cov.append(cov)
                                labels.append(neo_label)

                
                                count_novel=count_novel+1

                                if(count_novel-count)==10:
                                    for i in range(class_seen+count,class_seen+count_novel):
                                        cls_means.append(textEmbed[i])
                                    #print("len of cls_means after update",len(cls_means))
                                    #print("len of cls_cov after update",len(cls_cov))
                                    #print("len of labels after update",len(labels))
                                    print("count",count)
                                    print("count novel", count_novel)
                                    count=count_novel
                                    neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                                    textEmbed=neo_protos
                            

                        

                
                            
    variable_threshold=avg_threshold-k*std_threshold
    print("variable threshold for old classes:",variable_threshold[:class_seen])
    print("variable threshold for novel classes",variable_threshold[class_seen:])
    print("textEmbed shape",textEmbed.size())
    print("old",old)
    print("seen",seen)
    print("unseen",unseen)
    print("out of budget",out_of_bud)
    print("budget",budget)
    print("confusion buffer length",len(confusing_buffer))
    print("confusing label",confusing_label)
    print("textEmbedNovel",textEmbedNovel)
    preds=torch.Tensor(preds)
    #print("preds shape",preds.size())
    preds = preds.cpu().numpy()
    print("target",len(targets))
    print("preds",len(preds))
    print(len(list(set(preds))), len(preds))

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc

#top3 with confusion buffer

def test_on_the_fly_active3(model,dino_head,projection_head, cls_means,cls_cov,labels,test_loader
                        ,avg_threshold,std_threshold,k,save_name,args):

    textEmbed=projection_head.protos
    #print("textEmbed",textEmbed.size())
    model.eval()
    dino_head.eval()
    print("k:",k)
    #variable_threshold_txt=variable_threshold_txt.to(device)
    avg_threshold=avg_threshold.to(device)
    std_threshold=std_threshold.to(device)
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
        feats,logits = dino_head(feats)
        feats = F.normalize(feats, dim=-1)[:, :]
        #print(feats.shape)
        feats=feats.cpu().detach().numpy()
        all_feats.append(feats)
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # On-The-Fly
    # -----------------------
    
    cls_cov=cls_cov.tolist()
    cls_means=cls_means.tolist()
    labels=labels.tolist()
    print("len of cls_cov before testing",len(cls_cov))
    print("len of cls_means before testing",len(cls_means))
    print("len of labels before testing",len(labels))


    textEmbed=textEmbed.to(torch.float32)
    textEmbed=textEmbed.to(device)
    class_seen=textEmbed.size(0)
    #print(class_seen)
    
    all_feats = np.concatenate(all_feats)
    #print("all_feats",all_feats.shape)
    feat_list=all_feats.tolist()
    #print("target: ", targets[0:2])
    preds = []
    confusing_label={}
    textEmbedNovel={}
    for i in range(20*textEmbed.size(0)):
        textEmbedNovel[i]=[]
        confusing_label[i]=[]
    confusing_buffer=[]
    count_novel=0
    count=0
    old=0
    seen=0
    bud=0
    out_of_bud=0
    budget=2.5*class_seen
    count_sample=-1
    unseen=0
    flag=0
    for feat in feat_list:
        #print(feat)
        feat=torch.Tensor(feat).to(device)
        #print("feat shape ",feat.size())
        count_sample=count_sample+1
        cos_sim= feat @ textEmbed.T
        #print("cos_sim size",cos_sim.size())
        variable_threshold=avg_threshold-k*std_threshold
        diff_text=cos_sim - variable_threshold
        #diff_image=mean_cos_sim -  variable_threshold_image
        #print("diff:",diff)
        thres_val_text,index_text =  torch.topk(diff_text, dim=-1, largest=True, k=1)
        #thres_val_image,index_image =  torch.topk(diff_image, dim=-1, largest=True, k=1)
        _,pred_index =  torch.topk(diff_text, dim=-1, largest=True, k=3)
        
        index0=pred_index[0]
        index1=pred_index[1]
        index2=pred_index[2]

        if thres_val_text>=0:
            
                preds.append(index_text)
        else:
            if len(confusing_buffer)>=2:
                #print("Number of confusing samples collected:",len(confusing_samples))
                confusing_samples= torch.stack(confusing_buffer,dim=0)
                confusion_cos_sim=feat @ confusing_samples.T
                #print("confusion cos sim",confusion_cos_sim)
                confusing_threshold=computeConfusingThreshold(variable_threshold,confusing_label,confusing_buffer)
                #print("confusing threshold",confusing_threshold)
                con_diff_text=confusion_cos_sim - confusing_threshold
                con_val,con_index =  torch.topk(con_diff_text, dim=-1, largest=True, k=1)
                if con_val>=0:
                    temp=confusing_label[con_index.item()]
                    temp=torch.tensor(temp).item()
                    #print("temp",temp)
                    preds.append(temp)
                else:
                    if(bud<budget):
                        bud=bud+1
                        feat_label=targets[count_sample]
                        feat_label= int(feat_label)
                        #print("feat label",feat_label)
                        flag=0
                        flag_seen=0
                        flag_unseen=0
                        for i in range(pred_index.size(0)):
                    
                            index0=(pred_index[i]).item()
                            #print("pred index",index0)
                            if index0<class_seen:
                        
                                if (index0 == feat_label):
                                    preds.append(feat_label)
                                    
                                    confusing_buffer.append(feat)
                                    con_len=len(confusing_buffer)-1
                                    confusing_label[con_len].append(feat_label)
                                    flag=1
                                    seen=seen+1
                                    flag_seen=1
                                    break
                            #seen=seen+1
                            elif index0>=class_seen :
                                #print("pred index",index0)
                                id=index0-class_seen
                                #print("pred_index-class_seen",id)
                        
                                temp=torch.tensor(textEmbedNovel[id]).item()
                                #print(temp)
                                #print(textEmbedNovel)
                                if(feat_label==temp):
                                    preds.append(index0)
                                    
                                    confusing_buffer.append(feat)
                                    con_len=len(confusing_buffer)-1
                                    confusing_label[con_len].append(index0)
                                    flag=1
                                    old=old+1
                                    flag_unseen=1
                                    break
                        #print("flag",flag)
                        #print("flag seen",flag_seen)
                        #print("flag_unseen",flag_unseen)    
                        if(flag!=1):
                            #print("hi from flag")
                            unseen=unseen+1
                            feat=feat.unsqueeze(0)
                            #print(feat.size())
                            #print(thres_val)
                            textEmbed=torch.cat([textEmbed,feat],dim=0)
                
                            new_threshold_avg=avg_threshold[index_text]
                            new_threshold_std=std_threshold[index_text]
                    
                            #print("new_threshold:",new_threshold)
                            avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                            std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                            neo_label=textEmbed.size(0)
                            preds.append(neo_label)
                            #print("feat_label",feat_label)
                            novel=neo_label-class_seen-1
                            #print("novel",novel)
                            textEmbedNovel[novel].append(feat_label)
                            #print(textEmbedNovel)
                            id=random.randint(0, class_seen-1)
                            cov=cls_cov[id]
                            cls_cov.append(cov)
                            labels.append(neo_label)
                            count_novel=count_novel+1
                            if(count_novel-count)==50:
                                for i in range(class_seen+count,class_seen+count_novel):
                                    cls_means.append(textEmbed[i])
                                #print("len of cls_means after update",len(cls_means))
                                #print("len of cls_cov after update",len(cls_cov))
                                #print("len of labels after update",len(labels))
                                print("count",count)
                                print("count novel", count_novel)
                                count=count_novel
                                neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                                textEmbed=neo_protos

                    elif bud >= budget :
                        unseen=unseen+1
                        out_of_bud=out_of_bud+1
                        feat=feat.unsqueeze(0)
                        feat_label=targets[count_sample]
                        feat_label= int(feat_label)
                        #print(feat.size())
                        #print(thres_val)
                        textEmbed=torch.cat([textEmbed,feat],dim=0)

                        new_threshold_avg=avg_threshold[index_text]
                        new_threshold_std=std_threshold[index_text]

        
                        #print("new_threshold:",new_threshold)
                        avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                        std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                        neo_label=textEmbed.size(0)
                        preds.append(neo_label)
                        #print("feat_label",feat_label)
                        novel=neo_label-class_seen-1
                        #print("novel",novel)
                        textEmbedNovel[novel].append(feat_label)
                        #print(textEmbedNovel)
                        id=random.randint(0, class_seen-1)
                        cov=cls_cov[id]
                        cls_cov.append(cov)
                        labels.append(neo_label)

                
                        count_novel=count_novel+1

                        if(count_novel-count)==50:
                            for i in range(class_seen+count,class_seen+count_novel):
                                cls_means.append(textEmbed[i])
                            #print("len of cls_means after update",len(cls_means))
                            #print("len of cls_cov after update",len(cls_cov))
                            #print("len of labels after update",len(labels))
                            print("count",count)
                            print("count novel", count_novel)
                            count=count_novel
                            neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                            textEmbed=neo_protos
                            
            elif len(confusing_buffer)<2:
                    if(bud<budget):
                        bud=bud+1
                        feat_label=targets[count_sample]
                        feat_label= int(feat_label)
                        #print("feat label",feat_label)
                        flag=0
                        flag_seen=0
                        flag_unseen=0
                        for i in range(pred_index.size(0)):
                    
                            index0=(pred_index[i]).item()
                            #print("pred index",index0)
                            if index0<class_seen:
                        
                                if (index0 == feat_label):
                                    preds.append(feat_label)
                                    
                                    confusing_buffer.append(feat)
                                    con_len=len(confusing_buffer)-1
                                    confusing_label[con_len].append(feat_label)
                                    flag=1
                                    seen=seen+1
                                    flag_seen=1
                                    break
                            #seen=seen+1
                            elif index0>=class_seen :
                                #print("pred index",index0)
                                id=index0-class_seen
                                #print("pred_index-class_seen",id)
                        
                                temp=torch.tensor(textEmbedNovel[id]).item()
                                #print(temp)
                                #print(textEmbedNovel)
                                if(feat_label==temp):
                                    preds.append(index0)
                                    
                                    confusing_buffer.append(feat)
                                    con_len=len(confusing_buffer)-1
                                    confusing_label[con_len].append(index0)
                                    flag=1
                                    old=old+1
                                    flag_unseen=1
                                    break
                        #print("flag",flag)
                        #print("flag seen",flag_seen)
                        #print("flag_unseen",flag_unseen)    
                        if(flag!=1):
                            #print("hi from flag")
                            unseen=unseen+1
                            feat=feat.unsqueeze(0)
                            #print(feat.size())
                            #print(thres_val)
                            textEmbed=torch.cat([textEmbed,feat],dim=0)
                
                            new_threshold_avg=avg_threshold[index_text]
                            new_threshold_std=std_threshold[index_text]
                    
                            #print("new_threshold:",new_threshold)
                            avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                            std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                            neo_label=textEmbed.size(0)
                            preds.append(neo_label)
                            #print("feat_label",feat_label)
                            novel=neo_label-class_seen-1
                            #print("novel",novel)
                            textEmbedNovel[novel].append(feat_label)
                            #print(textEmbedNovel)
                            id=random.randint(0, class_seen-1)
                            cov=cls_cov[id]
                            cls_cov.append(cov)
                            labels.append(neo_label)
                            count_novel=count_novel+1
                            if(count_novel-count)==50:
                                for i in range(class_seen+count,class_seen+count_novel):
                                    cls_means.append(textEmbed[i])
                                #print("len of cls_means after update",len(cls_means))
                                #print("len of cls_cov after update",len(cls_cov))
                                #print("len of labels after update",len(labels))
                                print("count",count)
                                print("count novel", count_novel)
                                count=count_novel
                                neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                                textEmbed=neo_protos

                    elif bud>=budget:
                        feat_label=targets[count_sample]
                        feat_label= int(feat_label)
                        unseen=unseen+1
                        out_of_bud=out_of_bud+1
                        feat=feat.unsqueeze(0)
                        #print(feat.size())
                        #print(thres_val)
                        textEmbed=torch.cat([textEmbed,feat],dim=0)

                        new_threshold_avg=avg_threshold[index_text]
                        new_threshold_std=std_threshold[index_text]

        
                        #print("new_threshold:",new_threshold)
                        avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                        std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                        neo_label=textEmbed.size(0)
                        preds.append(neo_label)
                        #print("feat_label",feat_label)
                        novel=neo_label-class_seen-1
                        #print("novel",novel)
                        textEmbedNovel[novel].append(feat_label)
                        #print(textEmbedNovel)
                        id=random.randint(0, class_seen-1)
                        cov=cls_cov[id]
                        cls_cov.append(cov)
                        labels.append(neo_label)

                
                        count_novel=count_novel+1

                        if(count_novel-count)==50:
                            for i in range(class_seen+count,class_seen+count_novel):
                                cls_means.append(textEmbed[i])
                            #print("len of cls_means after update",len(cls_means))
                            #print("len of cls_cov after update",len(cls_cov))
                            #print("len of labels after update",len(labels))
                            print("count",count)
                            print("count novel", count_novel)
                            count=count_novel
                            neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                            textEmbed=neo_protos
                                        
    variable_threshold=avg_threshold-k*std_threshold
    print("variable threshold for old classes:",variable_threshold[:class_seen])
    print("variable threshold for novel classes",variable_threshold[class_seen:])
    print("textEmbed shape",textEmbed.size())
    print("old",old)
    print("seen",seen)
    print("unseen",unseen)
    print("out of budget",out_of_bud)
    print("budget",budget)
    print("confusion buffer length",len(confusing_buffer))
    print("confusing label",confusing_label)
    print("textEmbedNovel",textEmbedNovel)
    preds=torch.Tensor(preds)
    #print("preds shape",preds.size())
    preds = preds.cpu().numpy()
    print("target",len(targets))
    print("preds",len(preds))
    print(len(list(set(preds))), len(preds))

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc




#TOP 3 predictions
def test_on_the_fly_active2(model,dino_head,projection_head, cls_means,cls_cov,labels,test_loader
                        ,avg_threshold,std_threshold,k,save_name,args):

    textEmbed=projection_head.protos
    #print("textEmbed",textEmbed.size())
    model.eval()
    dino_head.eval()
    print("k:",k)
    #variable_threshold_txt=variable_threshold_txt.to(device)
    avg_threshold=avg_threshold.to(device)
    std_threshold=std_threshold.to(device)
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
        feats,logits = dino_head(feats)
        feats = F.normalize(feats, dim=-1)[:, :]
        #print(feats.shape)
        feats=feats.cpu().detach().numpy()
        all_feats.append(feats)
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # On-The-Fly
    # -----------------------
    
    cls_cov=cls_cov.tolist()
    cls_means=cls_means.tolist()
    labels=labels.tolist()
    print("len of cls_cov before testing",len(cls_cov))
    print("len of cls_means before testing",len(cls_means))
    print("len of labels before testing",len(labels))


    textEmbed=textEmbed.to(torch.float32)
    textEmbed=textEmbed.to(device)
    class_seen=textEmbed.size(0)
    #print(class_seen)
    
    all_feats = np.concatenate(all_feats)
    #print("all_feats",all_feats.shape)
    feat_list=all_feats.tolist()
    #print("target: ", targets[0:2])
    preds = []
    textEmbedNovel={}
    for i in range(5*textEmbed.size(0)):
        textEmbedNovel[i]=[]
    count_novel=0
    count=0
    old=0
    seen=0
    bud=0
    out_of_bud=0
    budget=2.5*class_seen
    count_sample=-1
    unseen=0
    flag=0
    for feat in feat_list:
        #print(feat)
        feat=torch.Tensor(feat).to(device)
        #print("feat shape ",feat.size())
        count_sample=count_sample+1
        cos_sim= feat @ textEmbed.T
        #print("cos_sim size",cos_sim.size())
        variable_threshold=avg_threshold-k*std_threshold
        diff_text=cos_sim - variable_threshold
        #diff_image=mean_cos_sim -  variable_threshold_image
        #print("diff:",diff)
        thres_val_text,index_text =  torch.topk(diff_text, dim=-1, largest=True, k=1)
        _,pred_index =  torch.topk(diff_text, dim=-1, largest=True, k=3)
        
        index0=pred_index[0]
        index1=pred_index[1]
        index2=pred_index[2]


        if thres_val_text>=0:
            
                preds.append(index_text)
        else:

            if (bud < budget):
                bud=bud+1
                feat_label=targets[count_sample]
                feat_label= int(feat_label)
                #print("feat label",feat_label)
                flag=0
                flag_seen=0
                flag_unseen=0
                for i in range(pred_index.size(0)):
                    
                    index0=(pred_index[i]).item()
                    #print("pred index",index0)
                    if index0<class_seen:
                        
                        if (index0 == feat_label):
                            preds.append(feat_label)
                            flag=flag+1
                            seen=seen+1
                            flag_seen=1
                            break
                    #seen=seen+1
                    elif index0>=class_seen :
                        #print("pred index",index0)
                        id=index0-class_seen
                        #print("pred_index-class_seen",id)
                        
                        temp=torch.tensor(textEmbedNovel[id]).item()
                        #print(temp)
                        #print(textEmbedNovel)
                        if(feat_label==temp):
                            preds.append(index0)
                            flag=flag+1
                            old=old+1
                            flag_unseen=1
                            break
                #print("flag",flag)
                #print("flag seen",flag_seen)
                #print("flag_unseen",flag_unseen)    
                if(flag!=1):
                    #print("hi from flag")
                    unseen=unseen+1
                    feat=feat.unsqueeze(0)
                    #print(feat.size())
                    #print(thres_val)
                    textEmbed=torch.cat([textEmbed,feat],dim=0)
                
                    new_threshold_avg=avg_threshold[index_text]
                    new_threshold_std=std_threshold[index_text]
                    
                    #print("new_threshold:",new_threshold)
                    avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                    std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                    neo_label=textEmbed.size(0)
                    preds.append(neo_label)
                    #print("feat_label",feat_label)
                    novel=neo_label-class_seen-1
                    #print("novel",novel)
                    textEmbedNovel[novel].append(feat_label)
                    #print(textEmbedNovel)
                    id=random.randint(0, class_seen-1)
                    cov=cls_cov[id]
                    cls_cov.append(cov)
                    labels.append(neo_label)
                    count_novel=count_novel+1
                    if(count_novel-count)==10:
                        for i in range(class_seen+count,class_seen+count_novel):
                            cls_means.append(textEmbed[i])
                        print("len of cls_means after update",len(cls_means))
                        print("len of cls_cov after update",len(cls_cov))
                        print("len of labels after update",len(labels))
                        print("count",count)
                        print("count novel", count_novel)
                        count=count_novel
                        neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                        textEmbed=neo_protos
            else:
                    out_of_bud=out_of_bud+1
                    feat=feat.unsqueeze(0)
                    #print(feat.size())
                    #print(thres_val)
                    textEmbed=torch.cat([textEmbed,feat],dim=0)
                
                    new_threshold_avg=avg_threshold[index_text]
                    new_threshold_std=std_threshold[index_text]
                    
                    #print("new_threshold:",new_threshold)
                    avg_threshold=torch.cat([avg_threshold,new_threshold_avg])
                    std_threshold=torch.cat([std_threshold,new_threshold_std])
                    

                    neo_label=textEmbed.size(0)
                    preds.append(neo_label)
                    #print("feat_label",feat_label)
                    novel=neo_label-class_seen-1
                    #print("novel",novel)
                    textEmbedNovel[novel].append(feat_label)
                    #print(textEmbedNovel)
                    id=random.randint(0, class_seen-1)
                    cov=cls_cov[id]
                    cls_cov.append(cov)
                    labels.append(neo_label)
                    count_novel=count_novel+1
                    if(count_novel-count)==10:
                        for i in range(class_seen+count,class_seen+count_novel):
                            cls_means.append(textEmbed[i])
                        print("len of cls_means after update",len(cls_means))
                        print("len of cls_cov after update",len(cls_cov))
                        print("len of labels after update",len(labels))
                        print("count",count)
                        print("count novel", count_novel)
                        count=count_novel
                        neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                        textEmbed=neo_protos
    variable_threshold=avg_threshold-k*std_threshold
    print("variable threshold for old classes:",variable_threshold[:class_seen])
    print("variable threshold for novel classes",variable_threshold[class_seen:])
    print("textEmbed shape",textEmbed.size())
    print("old",old)
    print("seen",seen)
    print("unseen",unseen)
    
    print("budget",budget)
    print("out of budget",out_of_bud)

    preds=torch.Tensor(preds)
    #print("preds shape",preds.size())
    preds = preds.cpu().numpy()
    print("target",len(targets))
    print("preds",len(preds))
    print(len(list(set(preds))), len(preds))

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc


def test_on_the_fly(model,dino_head, projection_head, test_loader,epoch,avg_threshold,std_threshold,k,save_name,args):
    model.eval()
    dino_head.eval()
    print("k:",k)
    textEmbed=projection_head.protos
    avg_threshold=avg_threshold.to(device)
    std_threshold=std_threshold.to(device)
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
        feats,logits = dino_head(feats)

        feats = F.normalize(feats, dim=-1)[:, :]
#         print(feats.shape)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # On-The-Fly
    # -----------------------
   
    textEmbed=textEmbed.to(torch.float32)
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

        

def test_on_the_fly_CA(model,dino_head,projection_head, cls_means,cls_cov,labels,test_loader
                        ,avg_threshold,std_threshold,k,save_name,args):

    textEmbed=projection_head.protos
    #print("textEmbed",textEmbed.size())
    model.eval()
    dino_head.eval()
    print("k:",k)
    #variable_threshold_txt=variable_threshold_txt.to(device)
    avg_threshold=avg_threshold.to(device)
    std_threshold=std_threshold.to(device)
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
        feats,logits = dino_head(feats)
        feats = F.normalize(feats, dim=-1)[:, :]
        #print(feats.shape)
        feats=feats.cpu().detach().numpy()
        all_feats.append(feats)
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # On-The-Fly
    # -----------------------
    
    cls_cov=cls_cov.tolist()
    cls_means=cls_means.tolist()
    labels=labels.tolist()
    print("len of cls_cov before testing",len(cls_cov))
    print("len of cls_means before testing",len(cls_means))
    print("len of labels before testing",len(labels))


    textEmbed=textEmbed.to(torch.float32)
    textEmbed=textEmbed.to(device)
    class_seen=textEmbed.size(0)
    #print(class_seen)
    
    all_feats = np.concatenate(all_feats)
    #print("all_feats",all_feats.shape)
    feat_list=all_feats.tolist()
    #print("target: ", targets[0:2])
    preds = []
    count_novel=0
    count=0
    for feat in feat_list:
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

                neo_label=textEmbed.size(0)
                preds.append(neo_label)

                id=random.randint(0, class_seen-1)
                cov=cls_cov[id]
                cls_cov.append(cov)
                labels.append(neo_label)

                
                count_novel=count_novel+1

                if(count_novel-count)==30:
                    for i in range(class_seen+count,class_seen+count_novel):
                        cls_means.append(textEmbed[i])
                    print("len of cls_means after update",len(cls_means))
                    print("len of cls_cov after update",len(cls_cov))
                    print("len of labels after update",len(labels))
                    print("count",count)
                    print("count novel", count_novel)
                    count=count_novel
                    neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                    textEmbed=neo_protos

                
                #preds.append(textEmbed.size(0))
            
    variable_threshold=avg_threshold-k*std_threshold
    print("variable threshold for old classes:",variable_threshold[:class_seen])
    print("variable threshold for novel classes",variable_threshold[class_seen:])
    print("textEmbed shape",textEmbed.size())
  
    preds=torch.Tensor(preds)
    #print("preds shape",preds.size())
    preds = preds.cpu().numpy()
    print("target",len(targets))
    print("preds",len(preds))
    print(len(list(set(preds))), len(preds))

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc


def test_on_the_fly_baseline(model,dino_head,projection_head, cls_means,cls_cov,labels,test_loader
                        ,avg_threshold,std_threshold,k,save_name,args):

    textEmbed=projection_head.protos
    #print("textEmbed",textEmbed.size())
    model.eval()
    dino_head.eval()
    print("k:",k)
    #variable_threshold_txt=variable_threshold_txt.to(device)
    avg_threshold=avg_threshold.to(device)
    std_threshold=std_threshold.to(device)
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
        feats,logits = dino_head(feats)
        feats = F.normalize(feats, dim=-1)[:, :]
        #print(feats.shape)
        feats=feats.cpu().detach().numpy()
        all_feats.append(feats)
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # On-The-Fly
    # -----------------------
    
    cls_cov=cls_cov.tolist()
    cls_means=cls_means.tolist()
    labels=labels.tolist()
#    print("len of cls_cov before testing",len(cls_cov))
#    print("len of cls_means before testing",len(cls_means))
    print("len of labels before testing",len(labels))


    textEmbed=textEmbed.to(torch.float32)
    textEmbed=textEmbed.to(device)
    class_seen=textEmbed.size(0)
    #print(class_seen)
    
    all_feats = np.concatenate(all_feats)
    #print("all_feats",all_feats.shape)
    feat_list=all_feats.tolist()
    #print("target: ", targets[0:2])
    preds = []
    textEmbedNovel={}
    for i in range(20*textEmbed.size(0)):
        textEmbedNovel[i]=[]
    
    count_novel=0
    count=0
    #old=0
    #seen=0
    bud=0
    out_of_bud=0
    budget=2.5*class_seen
    count_sample=-1
    unseen=0
    flag=0
    
    #count_novel=0
    #count=0
    for feat in feat_list:
        #print(feat)
        feat=torch.Tensor(feat).to(device)
        #print("feat shape ",feat.size())
        count_sample=count_sample+1
        cos_sim= feat @ textEmbed.T
        value,index=CosineSim(img_feat=feat,textEmbed=textEmbed.T)

        if value>0.7:
            preds.append(index)

        else:
            if (bud < budget):
                bud=bud+1
                feat_label=targets[count_sample]
                feat_label= int(feat_label)
                #print("feat label",feat_label)
                flag=0
                #print("pred index",index)

                if index<class_seen:
                        
                    if (index == feat_label):
                        preds.append(feat_label)
                        flag=flag+1

                elif index>=class_seen :
                    #print(textEmbedNovel)
                    id=(index-class_seen).item()
                    #print("pred_index-class_seen",id)
                    t=textEmbedNovel[id]
                    #print(t)
                    temp=torch.tensor(textEmbedNovel[id]).item()
                    #print(temp)
                    #print(textEmbedNovel)
                    if(feat_label==temp):
                        preds.append(index)
                        flag=flag+1
            
                #print("flag",flag)
                 
                if(flag!=1):
                    #print("hi from flag")
                    unseen=unseen+1
                    feat=feat.unsqueeze(0)
                    #print(feat.size())
                    #print(thres_val)
                    textEmbed=torch.cat([textEmbed,feat],dim=0)
                    neo_label=textEmbed.size(0)
                    preds.append(neo_label)
                    #print("feat_label",feat_label)
                    novel=neo_label-class_seen-1
                    #print("novel",novel)
                    textEmbedNovel[novel].append(feat_label)
                    #print(textEmbedNovel)
                    id=random.randint(0, class_seen-1)
                    cov=cls_cov[id]
                    cls_cov.append(cov)
                    labels.append(neo_label)
                    count_novel=count_novel+1
                    if(count_novel-count)==10:
                        for i in range(class_seen+count,class_seen+count_novel):
                            cls_means.append(textEmbed[i])
                        #print("len of cls_means after update",len(cls_means))
                        #print("len of cls_cov after update",len(cls_cov))
                        #print("len of labels after update",len(labels))
                        print("count",count)
                        print("count novel", count_novel)
                        count=count_novel
                        neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                        textEmbed=neo_protos
            else:
                    out_of_bud=out_of_bud+1
                    feat=feat.unsqueeze(0)
                    #print(feat.size())
                    #print(thres_val)
                    textEmbed=torch.cat([textEmbed,feat],dim=0)

                    neo_label=textEmbed.size(0)
                    preds.append(neo_label)
                    #print("feat_label",feat_label)
                    novel=neo_label-class_seen-1
                    #print("novel",novel)
                    textEmbedNovel[novel].append(feat_label)
                    #print(textEmbedNovel)
                    id=random.randint(0, class_seen-1)
                    cov=cls_cov[id]
                    cls_cov.append(cov)
                    labels.append(neo_label)
                    count_novel=count_novel+1
                    if(count_novel-count)==10:
                        for i in range(class_seen+count,class_seen+count_novel):
                            cls_means.append(textEmbed[i])
                        print("count",count)
                        print("count novel", count_novel)
                        count=count_novel
                        neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                        textEmbed=neo_protos
            
    #variable_threshold=avg_threshold-k*std_threshold
    #print("variable threshold for old classes:",variable_threshold[:class_seen])
    #print("variable threshold for novel classes",variable_threshold[class_seen:])
    print("textEmbed shape",textEmbed.size())
  
    preds=torch.Tensor(preds)
    #print("preds shape",preds.size())
    preds = preds.cpu().numpy()
    print("target",len(targets))
    print("preds",len(preds))
    print(len(list(set(preds))), len(preds))

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc


def test_on_the_fly_margin_baseline(model,dino_head,projection_head, cls_means,cls_cov,labels,test_loader
                        ,avg_threshold,std_threshold,k,save_name,args):

    textEmbed=projection_head.protos
    #print("textEmbed",textEmbed.size())
    model.eval()
    dino_head.eval()
    print("k:",k)
    #variable_threshold_txt=variable_threshold_txt.to(device)
    avg_threshold=avg_threshold.to(device)
    std_threshold=std_threshold.to(device)
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
        feats,logits = dino_head(feats)
        feats = F.normalize(feats, dim=-1)[:, :]
        #print(feats.shape)
        feats=feats.cpu().detach().numpy()
        all_feats.append(feats)
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # On-The-Fly
    # -----------------------
    
    cls_cov=cls_cov.tolist()
    cls_means=cls_means.tolist()
    labels=labels.tolist()
#    print("len of cls_cov before testing",len(cls_cov))
#    print("len of cls_means before testing",len(cls_means))
    print("len of labels before testing",len(labels))


    textEmbed=textEmbed.to(torch.float32)
    textEmbed=textEmbed.to(device)
    class_seen=textEmbed.size(0)
    #print(class_seen)
    #margin_threshold=margin_mean-3*margin_std
    all_feats = np.concatenate(all_feats)
    #print("all_feats",all_feats.shape)
    feat_list=all_feats.tolist()
    #print("target: ", targets[0:2])
    preds = []
    textEmbedNovel={}
    for i in range(100*textEmbed.size(0)):
        textEmbedNovel[i]=[]
    
    count_novel=0
    count=0
    #old=0
    #seen=0
    bud=0
    out_of_bud=0
    budget=2.5*class_seen
    count_sample=-1
    unseen=0
    flag=0
    
    #count_novel=0
    #count=0
    for feat in feat_list:
        #print(feat)
        feat=torch.Tensor(feat).to(device)
        #print("feat shape ",feat.size())
        count_sample=count_sample+1
        cos_sim= feat @ textEmbed.T
        value,index=CosineSimMargin(img_feat=feat,textEmbed=textEmbed.T)
        
        margin=value[0]-value[1]
        
        if margin>=0.0004:
            preds.append(index[0])

        else:
            print("value",value)
            print("margin",margin)
            #print("feat label",feat_label)
            if (bud < budget):
                bud=bud+1
                feat_label=targets[count_sample]
                feat_label= int(feat_label)
                #print("feat label",feat_label)
                flag=0
                #print("pred index",index)

                if index[0]<class_seen:
                        
                    if (index[0] == feat_label):
                        preds.append(feat_label)
                        flag=flag+1

                elif index[0]>=class_seen :
                    #print(textEmbedNovel)
                    id=(index[0]-class_seen).item()
                    #print("pred_index-class_seen",id)
                    t=textEmbedNovel[id]
                    #print(t)
                    temp=torch.tensor(textEmbedNovel[id]).item()
                    #print(temp)
                    #print(textEmbedNovel)
                    if(feat_label==temp):
                        preds.append(index[0])
                        flag=flag+1
            
                #print("flag",flag)
                 
                if(flag!=1):
                    #print("hi from flag")
                    unseen=unseen+1
                    feat=feat.unsqueeze(0)
                    #print(feat.size())
                    #print(thres_val)
                    textEmbed=torch.cat([textEmbed,feat],dim=0)
                    neo_label=textEmbed.size(0)
                    preds.append(neo_label)
                    #print("feat_label",feat_label)
                    novel=neo_label-class_seen-1
                    #print("novel",novel)
                    textEmbedNovel[novel].append(feat_label)
                    #print(textEmbedNovel)
                    id=random.randint(0, class_seen-1)
                    cov=cls_cov[id]
                    cls_cov.append(cov)
                    labels.append(neo_label)
                    count_novel=count_novel+1
                    if(count_novel-count)==50:
                        for i in range(class_seen+count,class_seen+count_novel):
                            cls_means.append(textEmbed[i])
                        #print("len of cls_means after update",len(cls_means))
                        #print("len of cls_cov after update",len(cls_cov))
                        #print("len of labels after update",len(labels))
                        print("count",count)
                        print("count novel", count_novel)
                        count=count_novel
                        neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                        textEmbed=neo_protos
            else:
                    out_of_bud=out_of_bud+1
                    feat=feat.unsqueeze(0)
                    #print(feat.size())
                    #print(thres_val)
                    textEmbed=torch.cat([textEmbed,feat],dim=0)

                    neo_label=textEmbed.size(0)
                    preds.append(neo_label)
                    #print("feat_label",feat_label)
                    novel=neo_label-class_seen-1
                    #print("novel",novel)
                    textEmbedNovel[novel].append(feat_label)
                    #print(textEmbedNovel)
                    id=random.randint(0, class_seen-1)
                    cov=cls_cov[id]
                    cls_cov.append(cov)
                    labels.append(neo_label)
                    count_novel=count_novel+1
                    if(count_novel-count)==50:
                        for i in range(class_seen+count,class_seen+count_novel):
                            cls_means.append(textEmbed[i])
                        print("count",count)
                        print("count novel", count_novel)
                        count=count_novel
                        neo_protos=realign(cls_means=cls_means,cls_cov=cls_cov,labels=labels,textEmbed=textEmbed,args=args)
                        textEmbed=neo_protos
            
    #variable_threshold=avg_threshold-k*std_threshold
    #print("variable threshold for old classes:",variable_threshold[:class_seen])
    #print("variable threshold for novel classes",variable_threshold[class_seen:])
    print("textEmbed shape",textEmbed.size())
  
    preds=torch.Tensor(preds)
    #print("preds shape",preds.size())
    preds = preds.cpu().numpy()
    print("target",len(targets))
    print("preds",len(preds))
    print(len(list(set(preds))), len(preds))

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name,
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
    parser.add_argument('--warmup_proj_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=12)
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
        device = torch.device('cuda:1')
        args = get_class_splits(args)
        print("baseline margin 0.0004 mean-k*std,k=3,updated after 50 novel classes,realign epochs= 5")
        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(args.unlabeled_classes)
        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.code_dim = 12
        args.mlp_out_dim = 384

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

            #-----------------------
            #DINO HEAD
            #-----------------------
            dino_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                   out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
           

            #state_dict = torch.load(pretrain_path, map_location='cpu')
            #model.load_state_dict(state_dict)

            if args.warmup_model_dir is not None:
                print(f'Loading weights from {args.warmup_model_dir}')
                model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
            if args.warmup_proj_dir is not None:
                print(f'Loading weights from {args.warmup_proj_dir}')
                dino_head.load_state_dict(torch.load(args.warmup_proj_dir, map_location='cpu'))
            dino_head.to(device)
            model.to(device)

           

            # ----------------------
            # HOW MUCH OF BASE MODEL TO FINETUNE
            # ----------------------
            for m in model.parameters():
                m.requires_grad = False
            #print(model)

            # Only finetune layers from block 'args.grad_from_block' onwards
            #for name, m in model.named_parameters():
            #    if 'block' in name:
            #        block_num = int(name.split('.')[1])
            #        if block_num >= args.grad_from_block:
            #            m.requires_grad = True

        #else:

        #    raise NotImplementedError

        # --------------------
        # CONTRASTIVE TRANSFORM
        # --------------------
        train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
        train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

        # --------------------
        # DATASETS
        
        # --------------------
        # DATASETS
        # --------------------
        
        train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_dataset,subsample_classes = get_datasets2(args.dataset_name,
                                                                                             train_transform,
                                                                                             test_transform,
                                                                                             args)
                                                                                            

        label_len = len(labelled_dataset)
        print('Length of the labelled dataset:',label_len)
        # --------------------
        # DATALOADERS
        # --------------------
        labelled_train_loader = DataLoader(labelled_dataset, num_workers=args.num_workers, batch_size=args.batch_size, 
                                  shuffle=True, drop_last=True)
        unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                          batch_size=args.batch_size, shuffle=False)
        #text Embedding

        textEmbed=pickle.load(open("sbertImagenet.p","rb"))
        textEmbed=textEmbed.T
        #print("text Embed: ",textEmbed.size())
        #subsample classes
        subsample_classes=subsample_classes.to(torch.long)
        textEmbed=TextEmbedLabel(subsample_classes,textEmbed)
        textEmbed,_ = [t for t in textEmbed.chunk(2)]
        textEmbed=textEmbed.to(torch.float32)
        #print("textEmbed size",textEmbed.size())
        # ----------------------
        # PROJECTION HEAD
        # ----------------------
        
        centroid=computeMean(model,dino_head,textEmbed,labelled_train_loader)
        projection_head = prototypeClassifier(centroid)
        projection_head.to(device)
        textEmbed=centroid
        #-----------------------
        #COMPUTE MEAN & VARIANCE
        # ----------------------
        cls_means,cls_cov,labels=compute_mean_var(model,dino_head,labelled_train_loader)
        #print("hi")
        # ----------------------
        # TRAIN
        # ----------------------
        train(model,dino_head,projection_head,cls_means,cls_cov,labels,labelled_train_loader,test_loader, unlabelled_train_loader,textEmbed, args)

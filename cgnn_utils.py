# Copyright (c) 2020, University of Pittsburgh. All rights reserved.

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

def make_data_loader(dataset_train,dataset_val,dataset_test, batch_size, cuda, gpu, world_size, rank):

    preprocess(dataset_train, cuda)
    preprocess(dataset_val, cuda)
    preprocess(dataset_test, cuda)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,num_replicas=world_size,rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,num_replicas=world_size,rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test,num_replicas=world_size,rank=rank)
    
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   collate_fn=collate, num_workers=0, pin_memory=True, 
                                                   sampler=train_sampler)

    val_dataloader = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size, 
                                                 shuffle=False,
                                                 collate_fn=collate, num_workers=0, pin_memory=True, 
                                                 sampler=val_sampler)

    test_dataloader = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size, 
                                                  shuffle=False,
                                                  collate_fn=collate, num_workers=0, pin_memory=True, 
                                                  sampler=test_sampler)
                                                  

    return train_dataloader, val_dataloader, test_dataloader

def collate(data):
    """
    Collate function
    """
    graphs, labels = map(list, zip(*data))
    batched = dgl.batch(graphs)
    labels = torch.LongTensor(labels)
    return batched, labels

def preprocess(dataset, cuda):

    for g, _ in dataset:
        g,_ = g.to('cuda:0'), _.to('cuda:0')
        for key_g, val_g in g.ndata.items():
            processed = g.ndata.pop(key_g)
            processed = processed.type('torch.FloatTensor')
            if cuda:
                processed = processed.cuda()
            g.ndata[key_g] = processed
        for key_g, val_g in g.edata.items():
            processed = g.edata.pop(key_g)
            processed = processed.type('torch.FloatTensor')
            if cuda:
                processed = processed.cuda()
            g.edata[key_g] = processed
            
            
def test(data_loader, model, loss_fcn):
    """
    Testing
    :param data_loader: (data.Dataloader)
    :param model: (Model)
    :param loss_fcn: (torch.nn loss)
    :return: loss, accuracy
    """
    model.eval()
    losses = []
    accuracies = []
    
    true_labels = []
    pred_labels = []
    
    for iter, (graphs, labels) in enumerate(data_loader):

        graphs = graphs.to('cuda')
        logits = model(graphs)
        labels = labels.to('cuda')
        loss = loss_fcn(logits, labels)
        losses.append(loss.item())

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        accuracies.append(correct.item() * 1.0 / len(labels))

    return np.mean(losses), np.mean(accuracies)      

import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

import sklearn
import pickle 
import numpy as np
import pickle as pkl
import os
import argparse
import pandas as pd

import sys
sys.path.insert(1,'gnn-min-example-master/')

from core.dataloader.constants import TRAIN_RATIO, TEST_RATIO
from core.utils import read_params
from core.model import Model

from cgnn_utils import make_data_loader, collate, preprocess

from google_drive_downloader import GoogleDriveDownloader as gdd

# Download the graphs and labels to prepare the data
gdd.download_file_from_google_drive(file_id='1ie0FKLU9-PnoFKE_fhfE3wpLpjO0l6lg', dest_path='./data/tf_train.csv', unzip=True)
gdd.download_file_from_google_drive(file_id='1_tZO6ooCdpwaOVouml3WEv17nvxhQO_Z', dest_path='./data/tf_val.csv', unzip=True)
gdd.download_file_from_google_drive(file_id='1GFqi7ElDBBWONqGTHifTH8pKeoB2Lxwi', dest_path='./data/tf_test.csv', unzip=True)
gdd.download_file_from_google_drive(file_id='10vIC1UZKy2JmKHk7Abc3mTZsl0kbKdri', dest_path='./data/train_data_invasive', unzip=True)
gdd.download_file_from_google_drive(file_id='1bf212GBLyfHhyVK8LFnzMoREbXiLO7pg', dest_path='./data/val_data_invasive', unzip=True)
gdd.download_file_from_google_drive(file_id='1-waHZ8k0-iqFXmFdF6ZNzUK5RSZWjFbd', dest_path='./data/test_data_invasive', unzip=True)

# Load the data files

train = pd.read_csv('data/tf_train.csv')
val   = pd.read_csv('data/tf_val.csv')
test  = pd.read_csv('data/tf_test.csv')

train_images = np.array(train['Image'])
train_labels = train['Label']
val_images = np.array(val['Image'])
val_labels = val['Label']
test_images = np.array(test['Image'])
test_labels = test['Label']

with open('data/train_data_invasive', 'rb') as config_dictionary_file:
    train_data = pickle.load(config_dictionary_file)
    
with open('data/val_data_invasive', 'rb') as config_dictionary_file:
    val_data = pickle.load(config_dictionary_file)

with open('data/test_data_invasive', 'rb') as config_dictionary_file:
    test_data = pickle.load(config_dictionary_file)

def main(gpu, args):
    
    ############################################################
    world_size = args.gpus * args.nodes
    rank = args.nr * args.gpus + gpu                         
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    ############################################################
    
    config_params = read_params(args.config_fpath)
    
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        
    SAVE_DIR = 'models'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'mlp-cgnn_v2_invasive.pt')
    
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
            
    best_valid_loss = float('inf')
    
    
    # Load the dataset
    print('*** Create data loader ***')
    dataloader, val_dataloader, test_dataloader = make_data_loader(train_data, val_data
                                                               ,test_data, batch_size=16, cuda=True, gpu=gpu, world_size=world_size, rank=rank)
    
    print('*** Create model ***')
    model = Model(config=config_params, verbose=True, cuda=cuda)
    
    print(cuda)
    if cuda==True:
        torch.cuda.set_device(args.gpu)
        
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # loss function
    loss_fcn = torch.nn.CrossEntropyLoss()
    
    # Start training
    print('*** Start training ***')
    step = 0
    model.train()
    
    # DDP
    model = model.to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model.to(args.gpu), device_ids=[args.gpu])
    
    losses = []
    for epoch in range(args.n_epochs):
        print("Epoch:", epoch)
        for iter, (graphs, labels) in enumerate(dataloader):
            
            # forward pass
            graphs = graphs.to('cuda')
            logits = model(graphs)
            labels = labels.to('cuda')
            # compute loss
            loss = loss_fcn(logits, labels)
            losses.append(loss.item())

            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # testing
            step += 1
            if step % args.eval_every == 0:
                val_loss, val_acc = test(val_dataloader, model, loss_fcn)
                print(
                    "Step {:05d} | Train loss {:.4f} | Over {} | Val loss {:.4f} |"
                    "Val acc {:.4f}".format(
                        step,
                        np.mean(losses),
                        len(losses),
                        val_loss,
                        val_acc,
                    ))
                model.train()
                
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)

                    
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument("--config_fpath",type=str, default = 'gnn-min-example-master/core/config/config_file_binary.json', required=False,help="Path to JSON configuration file.")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight-decay",type=float,default=5e-4,help="Weight for L2 loss")
    parser.add_argument("--n-epochs",type=int,default=10,help="number of epochs")
    parser.add_argument("--batch-size",type=int,default=16,help="batch size")
    parser.add_argument("--eval-every",type=int,default=50,help="eval model every N steps")
    
    # Data distributed parallel training params
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--gpus", type=int, default=1, help="gpus")
    parser.add_argument("--nodes", type=int, default=1, help="nodes")
    parser.add_argument("--nr", type=int, default=0, help="nr")
    
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8800' 
    mp.spawn(main(0,args), nprocs= gpus, args = Args())
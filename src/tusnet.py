# --------------------------------------------------------------------------- #
# TUSNet Training Script
# --------------------------------------------------------------------------- #
#                                                                             #
# This script trains TUSNet, a deep learning model for simulating             #
# transcranial ultrasound (TUS) pressure fields within a fraction of a        #
#                                                                             #
# Features:                                                                   #
# - Load a pre-trained model for fine-tuning or train from scratch.           #
# - Distributed training using multiple GPUs for efficiency.                  #
# - Configurable training modes and network types.                            #
#                                                                             #
# Example Usage (Training from Scratch):                                      #
# python3 tusnet.py --net field --lr 2e-4 --layers 1 --batch_size 256         #
#                                                                             #
# Example Usage (Transfer Learning):                                          #
# python3 tusnet.py --pretrained model.pth --lr 2e-4 --layers 4               #
#                   --batch_size 256 --net field  --trmode=transfer           #
#                                                                             #
# Arguments:                                                                  #
# --pretrained : Path to a pre-trained model checkpoint                       #
#                (for regression or transfer learning)                        #
# --lr         : Learning rate for the optimizer (def. 2e-4)                  #
# --layers     : Number of LSTM layers in the model architectur (def. 1)      #
# --batch_size : Batch size for training (def. 256)                           #
# --net        : Network component to train (def. field / encoder-decoder)    #
# --trmode     : Training mode (def. blank / train from scratch)              #
# --mode       : Test run or complete training (def. blank / complete)        #
#                                                                             #
# --------------------------------------------------------------------------- #

## ----- OS ----- ##
import os
import sys
import pytz
import time
import psutil
import random
import argparse
from natsort import os_sorted
from datetime import datetime

## ----- MATH / STATS ----- ##
import scipy.io
import numpy as np  
from copy import deepcopy
from collections import defaultdict

## ----- TORCH ----- ##
import gc
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset

## ----- FUNCTIONS ----- ##
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/models")
from functions import *

## ----- TUSNET ----- ##
from LSTMConvBlock import *
from LSTMConvCell import *
from PhaseDecoder import *
from PressureDecoder import *
from dataset import *

## ----- PLOTTING ----- ##
from PIL import Image
from IPython import display
from tqdm.auto import trange
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

## ----- REPRODUCIBILITY ----- ##
seed = 111
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

## ----- DISTRIBUTED DATA PARALLEL ----- ##
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


## ----- CREATE PROCESS GROUP ----- ##
def ddp_setup(rank, world_size):
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5000"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    if rank == 0:
        print("Process Groups Initialized!")
    
    
## ----- TRAINING FUNCTION ----- ##
def train_model(rank: int, world_size: int, objective, dataloaders,
                regularizer, num_epochs, lr, batch_size, net, num_layers, starting_point, trmode):
    
    ## ----- FIGURES ----- ##
    
    # Losses and Learning Rates #
    fig_path = "runs/" + str(datetime.now(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y, %H:%M")) + "/"
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(fig_path + "outputs/", exist_ok=True)
    os.makedirs(fig_path + "loss/", exist_ok=True)
    os.makedirs(fig_path + "models/", exist_ok=True)
    
    ## ----- TRAINING ----- ##
    
    # Encoding Block / Pressure Field
    train_losses = []
    val_losses = []
    train_images = []
    lrs = []
    model_weights_history = defaultdict(list)   # store some of the model weights
    model_grads_history = defaultdict(list)     # store somde of the parameter gradients

    # Decoding Block (DB)
    train_losses_db = []
    val_losses_db = []
    db_errors = []
    lrs_db = []
    
    test_batch = next(iter(dataloaders['val']))
    test_inputs = test_batch[0]
    test_outputs = from_t(test_batch[1])
    test_phases = from_t(test_batch[2])
    test_pressures = from_t(test_batch[3])
    
    torch.cuda.set_device(rank)
    gc.collect()
    torch.cuda.empty_cache()
    
    ## ----- DEFINE MODEL ----- ##
    
    if trmode == "regression":
        
        # Define model
        hidden_size = 512
        pfmodel = LSTMConvBlock(rank, num_cells=10, input_size=512, 
                                hidden_size=hidden_size, num_layers=num_layers)
        if net == "phase":
            model = PhaseDecoder(rank, ch_mult=1)
        elif net == "pressure":
            model = PressureDecoder(rank)
        
    else:
    
        # Define model
        hidden_size = 512
        model = LSTMConvBlock(rank, num_cells=10, input_size=512, 
                              hidden_size=hidden_size, num_layers=num_layers)
    
    ## ----- TRANSFER LEARNING ----- ##
        
    if starting_point != "":
    
        state_dict = torch.load(starting_point)
        
        if trmode == "regression":
            state_dict_new = pfmodel.state_dict()
        else:
            state_dict_new = model.state_dict()
        
        for param in state_dict:
            
            # Wrapping the model in DataParallel adds the 'module' prefix to the parameters
            if param[0:6] == "module":
                nparam = param[7:]
            else:
                nparam = param
                
            if trmode == "transfer":
                if 'conv1d.weight' in param:
                    state_dict_new[nparam][:1, :1, :] = state_dict[param]
                elif 'conv1d.bias' in param:
                    state_dict_new[nparam][:1] = state_dict[param]
                else:
                    state_dict_new[nparam] = state_dict[param]
            elif trmode == "continue" or trmode == "regression":
                state_dict_new[nparam] = state_dict[param]
                
        if trmode == "regression":
            pfmodel.load_state_dict(state_dict_new)
            pfmodel.eval()
            pfmodel = pfmodel.to(rank)
        else:
            model.load_state_dict(state_dict_new) 

    ## ----- DDP ----- ##
        
    model = model.to(rank)
    
    if trmode == "regression":
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # use standard SGD with a decaying learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr) #, weight_decay=1e-6)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                               factor=0.1, patience=2, 
                                               threshold=1e-3, verbose=False)
    
    scaler = torch.cuda.amp.GradScaler()
    
    ## ----- TRAIN ----- ##

    info = True # set to False for "silent" training
    
    # progress bars
    if rank == 0 and info:    
        
        print("")
        print("Memory Usage:", psutil.virtual_memory().percent, end = "\n\n")
        print("Parameter Count:", count_parameters(model.module), end = "\n\n")

        pbar = trange(num_epochs)
        pbar.set_description("Epoch")
        inner_pbar = trange(len(dataloaders['train'].dataset))
        inner_pbar.set_description("Batch")  
    
    for epoch in range(num_epochs): 
        
        if epoch < 5:
            lambda_skull, lambda_focus = 0.5, 0.5
        else:
            lambda_skull = 0.5 / (epoch - 5 + 1)
            lambda_focus = 1 - lambda_skull
            
        for phase in ['train', 'val']:
            
            # set epoch for DDP sampler
            dataloaders[phase].sampler.set_epoch(epoch)
            
            # set model to train/validation as appropriate
            if phase == 'train':
                model.train()
                if rank == 0 and info:
                    inner_pbar.reset()
            elif phase == 'val':
                model.eval()
            
            # track the running loss over batches
            running_loss = 0
            running_size = 0
            for i, data in enumerate(dataloaders[phase], 0):
                
                images = data[0].to(rank).float()
                if net == "field":
                    labels = data[1] #.to(rank).float()              
                elif net == "phase":
                    labels = data[2] #.to(rank).float()
                elif net == "pressure":
                    labels = data[3] #.to(rank).float()
                
                if phase == "train":
                    with torch.set_grad_enabled(True):
                        optimizer.zero_grad()
                        # compute the model output and loss
                        
                        if net == "phase" or net == "pressure":
                            h0 = torch.zeros(int(images.shape[0]), num_layers, 
                                             hidden_size).to(rank)
                            c0 = torch.zeros(int(images.shape[0]), num_layers, 
                                             hidden_size).to(rank)
                            with torch.no_grad():
                                _, encoded = pfmodel((images, h0, c0))
                                
                            with torch.cuda.amp.autocast():
                                output = model(encoded).squeeze()
                                loss = objective(output, labels)
                            
                        elif net == "field":    
                            h0 = torch.zeros(int(images.shape[0]), num_layers, 
                                             hidden_size).to(rank)
                            c0 = torch.zeros(int(images.shape[0]), num_layers, 
                                             hidden_size).to(rank)

                            with torch.cuda.amp.autocast():
                                output = model((images, h0, c0))[0].squeeze()
                                # loss = objective(output, labels)
                                
                                loss_skull = objective(output[:, 0:180, :], labels[:, 0:180, :]) 
                                loss_focus = objective(output[:, 180:, :], labels[:, 180:, :])
                                loss = (2 * lambda_skull * loss_skull) + (2 * lambda_focus * loss_focus)
                                
                        # only add the regularizer in the training phase
                        if regularizer is not None:
                            loss += regularizer(model)

                        # update the gradient
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                    if rank == 0 and info:
                        inner_pbar.update(world_size * batch_size)
                        if i % 20 == 0:
                            cuda_usage = torch.cuda.mem_get_info()
                            cuda_percentage = round(100 * cuda_usage[0] / cuda_usage[1])
                            inner_pbar.set_description("Batch {:03} Free Memory {:02}".format(i, cuda_percentage))
                        
                elif phase == "val":
                    # just compute the loss in validation
                    if net == "phase" or net == "pressure":
                        h0 = torch.zeros(int(images.shape[0]), num_layers, 
                                         hidden_size).to(rank)
                        c0 = torch.zeros(int(images.shape[0]), num_layers, 
                                         hidden_size).to(rank)
                        with torch.no_grad():
                            _, encoded = pfmodel((images, h0, c0))
                            output = model(encoded)

                        loss = objective(output, labels)
                    
                    elif net == "field":
                        h0 = torch.zeros(images.shape[0], num_layers, 
                                         hidden_size).to(rank)
                        c0 = torch.zeros(images.shape[0], num_layers, 
                                         hidden_size).to(rank)
                        with torch.no_grad():
                            output = model((images, h0, c0))[0].squeeze()
                            # loss = objective(output, labels)

                            loss_skull = objective(output[:, 0:180, :], labels[:, 0:180, :]) 
                            loss_focus = objective(output[:, 180:, :], labels[:, 180:, :])
                            loss = (2 * lambda_skull * loss_skull) + (2 * lambda_focus * loss_focus)

                assert torch.isfinite(loss)
                running_loss += loss.item()
                running_size += 1
                
            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet
            running_loss /= running_size
            if phase == 'train':
                if net == "phase" or net == "pressure":
                    train_losses_db.append(running_loss)
                elif net == "field":
                    train_losses.append(running_loss)
            else:
                if net == "phase" or net == "pressure":
                    val_losses_db.append(running_loss)
                elif net == "field":
                    val_losses.append(running_loss)
            
        # Update the learning rate
        scheduler.step(running_loss)

        if net == "phase" or net == "pressure":
            lrs_db.append(scheduler.optimizer.param_groups[0]['lr'])
        elif net == "field":
            lrs.append(scheduler.optimizer.param_groups[0]['lr'])
                
        # clear memory
        del images, labels, output, loss, h0, c0
                
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()
        
        ## ----- UPDATE DISPLAY ----- ## 

        if rank == 0 and info:
              
            inner_pbar.reset()
            pbar.update(1)

            if net == "field":

                pbar.set_description("Epoch {:03} Val {:.7f}"\
                                     .format(epoch, val_losses[-1]))

                ## Run a batch of inputs through the current network
                h0 = torch.zeros(len(test_inputs), num_layers, hidden_size).to(rank)
                c0 = torch.zeros(len(test_inputs), num_layers, hidden_size).to(rank) 
                with torch.no_grad():
                    tr_image = model((test_inputs, h0, c0))[0]

                ## Plot TUSNet output        
        
                plt.figure(figsize = (12, 4))
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.imshow(from_t(test_inputs[0]).squeeze(), cmap='gray')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
                plt.title("Input: Skull + Target")

                plt.subplot(1, 3, 2)
                plt.imshow(test_outputs[0].squeeze(), cmap='jet')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
                plt.title("True Output: Pressure Field")

                plt.subplot(1, 3, 3)
                plt.imshow(from_t(tr_image[0]).squeeze(), cmap='jet')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
                plt.title("Epoch " + str(epoch) + ' Output | Loss = ' + str(round(train_losses[-1], 5)))
                plt.tight_layout()
                plt.savefig(fig_path + "outputs/outputs_" + str(epoch) + ".png", dpi=150)
                plt.close()

                ## Plot current loss
                plt.figure(figsize = (12, 4))
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.plot(np.arange(0, len(train_losses), 1), train_losses, label='training')
                plt.plot(np.arange(0, len(val_losses), 1), val_losses, label='validation')
                plt.title("TUSNet Loss")
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")
                plt.yscale('log')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(np.arange(0, len(lrs), 1), lrs)
                plt.title("TUSNet LR")
                plt.xlabel("Epoch")
                plt.ylabel("Learning Rate")
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(fig_path + "loss/loss_" + str(epoch) + ".png", dpi=150)
                plt.close()

                del tr_image, h0, c0
                
            elif net == "phase" or net == "pressure":

                pbar.set_description("Epoch {:03} Val {:.7f}"\
                                     .format(epoch, val_losses_db[-1]))

                ## Run a batch of inputs through the current network
                h0 = torch.zeros(len(test_inputs), num_layers, hidden_size).to(rank)
                c0 = torch.zeros(len(test_inputs), num_layers, hidden_size).to(rank) 
                
                with torch.no_grad():
                    _, encoded = pfmodel((test_inputs, h0, c0))

                if net == "phase":

                    tr_image = from_t(model(encoded))
                    db_err = (test_phases - tr_image) / test_phases
                    db_errs = []
                    for i in range(np.size(db_err, 0)):
                        db_errs.append(np.mean(np.abs(db_err[np.isfinite(db_err)])))

                elif net == "pressure":

                    tr_image = from_t(model(encoded))
                    db_err = (test_pressures - tr_image) / test_pressures
                    db_errs = []
                    for i in range(np.size(db_err, 0)):
                        db_errs.append(np.mean(np.abs(db_err[np.isfinite(db_err)])))

                db_errors.append(100 * np.mean(db_errs))

                ## Plot TUSNet phase error      
                plt.figure(figsize = (12, 4))
                plt.clf()
                plt.plot(np.arange(0, len(db_errors), 1), db_errors, label='validation')
                plt.title("TUSNet " + net.capitalize() + " Arm Error")
                plt.xlabel("Epoch")
                plt.ylabel("Percentage Error (%)")
                plt.legend()
                plt.savefig(fig_path + "outputs/outputs_" + str(epoch) + ".png", dpi=150)
                plt.close()

                ## Plot current loss
                plt.figure(figsize = (12, 4))
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.plot(np.arange(0, len(train_losses_db), 1), train_losses_db, label='training')
                plt.plot(np.arange(0, len(val_losses_db), 1), val_losses_db, label='validation')
                plt.title("TUSNet " + net.capitalize() + " Arm Loss")
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")
                plt.yscale('log')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(np.arange(0, len(lrs_db), 1), lrs_db)
                plt.title("TUSNet " + net.capitalize() + " Arm LR")
                plt.xlabel("Epoch")
                plt.ylabel("Learning Rate")
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(fig_path + "loss/loss_" + str(epoch) + ".png", dpi=150)
                plt.close()

                del encoded, tr_image, h0, c0
                
        if epoch > 0 and epoch % 5 == 0:
            
            if net == "field":
                model_id = "pfmodel"
            elif net == "phase":
                model_id = "phasemodel"
            elif net == "pressure":
                model_id = "pressuremodel"
                
            # Save the trained model parameters
            torch.save(model.module.state_dict(), fig_path + "models/" + model_id + "_" + str(epoch) + ".pth")
            
    # Save the trained model parameters
    torch.save(model.module.state_dict(), fig_path + "models/" + model_id + "_latest.pth")
    
    return train_losses, val_losses, train_images
    
    
## ----- MAIN ----- ##    
def main(rank, world_size, batch_size, lr, layers, starting_point, trmode, net, mode):
    
    ## ----- PRINT MEMORY USAGE ----- ##

    # print("Rank:", rank)
    # print("Total Memory:", bytes_to_human_readable(psutil.virtual_memory().total))
    # print("Memory Usage:", psutil.virtual_memory().percent, end = "\n\n")
    
    torch.cuda.set_device(rank)
    ddp_setup(rank, world_size)
    
    ## ----- DATALOADER ----- ##
    
    path = "/home/shared/data/shards/"
    
    shard_paths = os_sorted(os.listdir(path + "inputs_traced/"))
    
    if mode == "development":
        shard_paths = shard_paths[0:20]
    
    shard_paths = [shard_path for shard_path in shard_paths if shard_path.endswith(".pth")]

    # Define the maximum number of shards to keep in memory
    max_cached_shards = 1

    # Create the shard cache
    train_shard_cache = ShardCache(max_cached_shards)

    # Create a list of dataset instances, one for each shard
    train_datasets = [CustomDataset(rank, path, shard_path, train_shard_cache) for shard_path in shard_paths]

    trainset = torch.utils.data.ConcatDataset(train_datasets)
    
    # Create a data loader that uses all the dataset instances
    trainloader = DataLoader(trainset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=8,
                             sampler=DistributedSampler(trainset, shuffle = False, drop_last = False))
    
    ## ----- DATALOADER ----- ##
    
    path = "/home/shared/data/shards_validation/"
    
    shard_paths = os_sorted(os.listdir(path + "inputs_traced/"))
    
    shard_paths = [shard_path for shard_path in shard_paths if shard_path.endswith(".pth")]

    # Define the maximum number of shards to keep in memory
    max_cached_shards = 1

    # Create the shard cache
    test_shard_cache = ShardCache(max_cached_shards)

    # Create a list of dataset instances, one for each shard
    test_datasets = [CustomDataset(rank, path, shard_path, test_shard_cache) for shard_path in shard_paths]

    testset = torch.utils.data.ConcatDataset(test_datasets)
    
    testloader = DataLoader(testset, 
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=8,
                            sampler=DistributedSampler(testset, shuffle = False, drop_last = False))

    # dataloaders for train and validation
    dataloaders = dict(train=trainloader, val=testloader)
    
    ## ----- TRAIN DECODER ARM ----- ##
    
    num_epochs = 20
    criterion = nn.L1Loss() # potentially only for phase and pressure decoders?
    # criterion = nn.MSELoss()

    world_size = torch.cuda.device_count()
    
    train_model(rank, world_size, criterion, dataloaders, None, num_epochs, lr, batch_size, net, layers, starting_point, trmode)
    
    destroy_process_group()

def parse_arguments():
    parser = argparse.ArgumentParser(description="TUSNet main training script")
    
    # Pre-trained model
    parser.add_argument("--pretrained", type=str, default="", 
                        help="Path to the pre-trained model checkpoint for transfer learning. Leave blank for training from scratch.")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=2e-4, 
                        help="Learning rate")
    parser.add_argument("--layers", type=int, default=1, 
                        help="Number of layers in the model you want to train.")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size for training.")
    
    # Model options
    parser.add_argument("--net", type=str, default = "field",
                        choices=["field", "phase", "pressure"], 
                        help="Network component to train [pressure field / encoder-decoder, phase vector decoder, or absolute pressure decoder]")
    parser.add_argument("--trmode", type=str, default = "",
                        choices=["", "regression", "continue", "transfer"], 
                        help="Training mode train [train fresh model (blank), continue training from a checkpoint ('continue'), transfer learn from a smaller (e.g. 1L) model ('transfer'), or train the phase/pressure decoder arms ('regression')")
    parser.add_argument("--mode", type=str, default = "".
                        choices=["", "development"], 
                        help="Training model [small test with 5,000 samples ('development') or complete training (blank)")
    
    return parser.parse_args()

if __name__ == "__main__":
    
    # Pre-Trained Model
    if len(sys.argv) > 1 and os.path.isfile(str(sys.argv[1])):
        starting_point = str(sys.argv[1])
        print("Pre-Trained Model Location:", starting_point, end="\n\n")
    else:
        starting_point = ""
        
    mode = ""
    net = "pressure"
    trmode = "regression"
    
    lr = 2e-4
    n_layers = 1
    batch_size = 256
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, batch_size, lr, n_layers, starting_point, trmode, net, mode), nprocs = world_size)
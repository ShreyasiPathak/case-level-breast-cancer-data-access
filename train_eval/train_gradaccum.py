# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:43:24 2021

@author: PathakS
"""

import re
import os
import math
import time
import torch
import datetime
import argparse
import random

import csv
import glob
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import metrics
from openpyxl import Workbook

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

from train_eval.test import test
from train_eval.optimization import optimizer_fn, select_lr_scheduler
from train_eval.loss_function import class_imbalance, loss_fn_bce, loss_fn_crossentropy, loss_fn_gmic, loss_fn_gmic_initialize
from train_eval.evaluation import results_store_excel, write_results_xlsx, write_results_xlsx_confmat, conf_mat_create

from models.sil_mil_model import SILmodel, MILmodel
from models.wu_resnet import SplitBreastModel

from utilities.pytorchtools import EarlyStopping, ModelCheckpoint
from utilities.utils import load_model, layer_selection_for_training, save_model
from utilities.dynamic_training_utils import dynamic_training

#import mlflow

#from torchviz import make_dot

#import tensorboard_log

#pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

#os.environ["PATH"] += os.pathsep + "/homes/spathak/.conda/envs/pytorch-env/lib/python3.9/site-packages/graphviz/dot"

def set_random_seed(config_params):
    #random state initialization of the code - values - 8, 24, 30
    torch.manual_seed(config_params['randseedother']) 
    torch.cuda.manual_seed(config_params['randseedother'])
    torch.cuda.manual_seed_all(config_params['randseedother'])
    np.random.seed(config_params['randseeddata'])
    random.seed(config_params['randseeddata'])
    g = torch.Generator()
    g.manual_seed(config_params['randseedother'])
    torch.backends.cudnn.deterministic = True
    return g

def model_initialization(config_params):
    if config_params['learningtype'] == 'SIL':
        model = SILmodel(config_params)
    elif config_params['learningtype'] == 'MIL':
        model = MILmodel(config_params)
    elif config_params['learningtype'] == 'MV':
        model = SplitBreastModel(config_params)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(f"{name} -> {param.device}", flush=True)
    #        #print(name)
    
    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(model))
    #mlflow.log_artifact("model_summary.txt")

    #print(model)
    if config_params['device']=='cuda':
        #cuda_device_list=list(map(int, config_params['device'].split(':')[1].split(',')))
        model = nn.DataParallel(model, device_ids = [0,1])
    model.to(torch.device(config_params['device']))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total model parameters:", pytorch_total_params, flush=True)
    count_param=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} -> {param.device}", flush=True)
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    return model, pytorch_total_params

def model_checkpoint(config_params, path_to_model):
    if config_params['patienceepochs']:
        modelcheckpoint = EarlyStopping(path_to_model=path_to_model, early_stopping_criteria=config_params['early_stopping_criteria'], patience=config_params['patienceepochs'], verbose=True)
    elif config_params['usevalidation']:
        modelcheckpoint = ModelCheckpoint(path_to_model=path_to_model, criteria=config_params['early_stopping_criteria'], verbose=True)
    return modelcheckpoint

def train(config_params, model, path_to_model, data_iterator_train, data_iterator_val, batches_train, batches_val, df_train, df_val, dataloader_test, batches_test, df_test, path_to_results_xlsx, path_to_results_text):
    '''Training'''
    if config_params['usevalidation']:
        modelcheckpoint = model_checkpoint(config_params, path_to_model)
    if config_params['trainingmethod'] == 'cosineannealing_pipnet':
        optimizer, optimizer_classifier = optimizer_fn(config_params, model)
        scheduler, scheduler_classifier = select_lr_scheduler(config_params, [optimizer, optimizer_classifier], batches_train)
        lrs_classifier = []
        lrs_scheduler = []
    else:
        optimizer = optimizer_fn(config_params, model)
        scheduler = select_lr_scheduler(config_params, optimizer, batches_train)
        lrs_scheduler = []
    class_weights_train = class_imbalance(config_params, df_train)

    if os.path.isfile(path_to_model):
        model, optimizer, start_epoch = load_model(model, optimizer, path_to_model)
        if config_params['patienceepochs']:
            modelcheckpoint = EarlyStopping(path_to_model=path_to_model, best_score=config_params['valloss_resumetrain'], early_stopping_criteria=config_params['early_stopping_criteria'], patience=config_params['patienceepochs'], verbose=True)
        print("start epoch:",start_epoch)
        print("lr:",optimizer.param_groups[0]['lr'])
    else:
        start_epoch = 0
        
    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss, bceloss = loss_fn_gmic_initialize(config_params, class_weights_train, test_bool=False)
    else:
        if config_params['activation'] == 'softmax':
            lossfn = loss_fn_crossentropy(config_params, class_weights_train, test_bool=False)
        elif config_params['activation'] == 'sigmoid':
            lossfn = loss_fn_bce(config_params, class_weights_train, test_bool=False)
    
    avg_mom_img = []
    avg_mom_side = []

    for epoch in range(start_epoch,config_params['maxepochs']):
        model.train()
        loss_train=0.0
        correct_train=0
        conf_mat_train=np.zeros((config_params['numclasses'],config_params['numclasses']))
        total_images_train=0
        batch_no=0
        eval_mode = False

        if config_params['trainingmethod'] == 'multisteplr1':
            model = layer_selection_for_training(model,epoch, config_params['trainingmethod'], epoch_step=5)
        
        count_param=0
        for name, param in model.named_parameters():
            if param.requires_grad:
                count_param+=1           
        print("Number of parameters that require gradient: ", count_param, flush=True)

        '''case_size = []
        case_grad_dic_img = dict()
        case_grad_dic_side = dict()
        case_mom_size = [] 
        case_mom_dic_img = dict() 
        case_mom_dic_side = dict()

        wb = Workbook()
        sheet1 = wb.active
        sheet1.title="results"
        header = ['BatchNum','ViewName','AvgMomImg','AvgMomSide']
        sheet1.append(header)
        wb.save(os.path.join(config_params['path_to_output'], 'average_mom_epoch'+str(epoch)+'.xlsx'))

        wb1 = Workbook()
        sheet2 = wb1.active
        sheet2.title="results"
        header2 = ['BatchNum','ViewName','AvgGradImg','AvgGradSide']
        sheet2.append(header2)
        wb1.save(os.path.join(config_params['path_to_output'], 'average_grad_epoch'+str(epoch)+'.xlsx'))
        '''

        for train_idx, train_batch1, train_labels1, views_names in data_iterator_train:
            print('Current Time after one batch loading:', time.ctime(time.time()), flush = True)
            print("train batch:", train_batch1.shape, flush=True)
            print("views name:", views_names)

            if config_params['viewsinclusion'] == 'all' and ((config_params['extra'] == 'dynamic_training_async') or (config_params['extra'] == 'dynamic_training_sync') or (config_params['extra'] == 'dynamic_training_momentumupdate')):
                model, optimizer, state_before_optim, lr_before_optim = dynamic_training(config_params, views_names, model, optimizer, None, None, True)
            
            '''weights_before_backprop = []
            parameter_name=[]

            for name, param in model.named_parameters(): # loop the weights in the model before updating and store them
                parameter_name.append(name)
                weights_before_backprop.append(param.clone())
            '''

            optimizer.zero_grad()  # clear previous gradients, compute gradients of all variables wrt loss
            
            
            #grad accumulation code
            loss_batch = 0
            
            for start_batch_id in range(0, train_batch1.shape[0], 2):
                if (start_batch_id+2)>train_batch1.shape[0]:
                    train_batch = train_batch1[start_batch_id:]
                    train_labels = train_labels1[start_batch_id:]
                else:
                    train_batch = train_batch1[start_batch_id:start_batch_id+2]
                    train_labels = train_labels1[start_batch_id:start_batch_id+2]
            
                train_batch = train_batch.to(config_params['device'])
                train_labels = train_labels.to(config_params['device'])
                train_labels = train_labels.view(-1)
                #print("train batch:", train_batch.shape, flush=True)
                #print("views name:", views_names)

                #if config_params['viewsinclusion'] == 'all' and ((config_params['extra'] == 'dynamic_training_async') or (config_params['extra'] == 'dynamic_training_sync') or (config_params['extra'] == 'dynamic_training_momentumupdate')):
                #    model, optimizer, state_before_optim, lr_before_optim = dynamic_training_utils.dynamic_training(config_params, views_names, model, optimizer, None, None, True)
                
                if config_params['femodel'] == 'gmic_resnet18':
                    if config_params['learningtype'] == 'SIL':
                        output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _ = model(train_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                        output_patch = None
                    elif config_params['learningtype'] == 'MIL':
                        output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _, output_patch = model(train_batch, views_names, eval_mode)
                    
                    #print("local shape:", output_batch_local.shape)
                    #print("global shape:", output_batch_global.shape)
                    #print("fusion shape:", output_batch_fusion.shape)
                    
                    if config_params['activation'] == 'sigmoid':
                        output_batch_local = output_batch_local.view(-1)
                        output_batch_global = output_batch_global.view(-1)
                        output_batch_fusion = output_batch_fusion.view(-1)
                        train_labels = train_labels.float()
                        if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                            pred = torch.ge(output_batch_fusion, torch.tensor(0.5)).float()
                        else:
                            pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
                    
                    elif config_params['activation'] == 'softmax':
                        pred = output_batch_fusion.argmax(dim=1, keepdim=True)
                    loss = loss_fn_gmic(config_params, bcelogitloss, bceloss, output_batch_local, output_batch_global, output_batch_fusion, saliency_map, train_labels, class_weights_train, output_patch, test_bool=False)

                else:
                    if config_params['learningtype'] == 'SIL':
                        output_batch = model(train_batch, eval_mode)
                    elif config_params['learningtype'] == 'MIL':
                        output_batch = model(train_batch, views_names, eval_mode)
                    elif config_params['learningtype'] == 'MV':
                        output_batch = model(train_batch, views_names, eval_mode)
                    
                    if config_params['activation'] == 'sigmoid':
                        if len(output_batch.shape)>1:
                            output_batch = output_batch.squeeze(1)
                        output_batch = output_batch.view(-1)                                                                    
                        train_labels = train_labels.float()
                        if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                            pred = torch.ge(output_batch, torch.tensor(0.5)).float()
                        else:
                            pred = torch.ge(torch.sigmoid(output_batch), torch.tensor(0.5)).float()
                        if config_params['classimbalance'] == 'focalloss':
                            loss = sigmoid_focal_loss(output_batch, train_labels, alpha=-1, reduction='mean')
                        else:
                            if config_params['classimbalance'] == 'poswt':
                                if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                                    weight_batch = torch.tensor([1, class_weights_train[0]]).to(config_params['device'])[train_labels.long()]
                                    lossfn.weight = weight_batch
                                    #print(weight_batch)
                            
                            loss = lossfn(output_batch, train_labels)
                    
                    elif config_params['activation'] == 'softmax':
                        pred = output_batch.argmax(dim=1, keepdim=True)
                        loss = lossfn(output_batch, train_labels)

                loss_train+=(train_labels.size()[0]*loss.item())

                '''weights_before_backprop = []
                parameter_name=[]

                for name, param in model.named_parameters(): # loop the weights in the model before updating and store them
                    parameter_name.append(name)
                    weights_before_backprop.append(param.clone())
                '''

                #optimizer.zero_grad()  # clear previous gradients, compute gradients of all variables wrt loss
                if config_params['trainingmethod'] == 'cosineannealing_pipnet':
                    optimizer_classifier.zero_grad()    
                
                
                #grad accumulation code
                loss = (train_batch.shape[0]*loss)/train_batch1.shape[0]
                loss_batch+=loss.item()
                
                loss.backward()

                #make_dot(loss, params=dict(model.named_parameters())).render("attached", format="png")
                
                #optimizer.step() # performs updates using calculated gradients
                if config_params['trainingmethod'] == 'cosineannealing_pipnet':
                    optimizer_classifier.step()

                #switch on the following part for consine annealing pipnet training method for blackbox.
                '''if scheduler!=None: 
                    scheduler.step()
                    print("scheduler lr:",scheduler.get_last_lr()[0], flush=True)
                    lrs_scheduler.append(scheduler.get_last_lr()[0])
                    if config_params['trainingmethod'] == 'cosineannealing_pipnet':
                        scheduler_classifier.step()
                        print("scheduler lr:",scheduler_classifier.get_last_lr()[0], flush=True)
                        lrs_classifier.append(scheduler_classifier.get_last_lr()[0])
                '''

                #batch_no=batch_no+1

                #if config_params['viewsinclusion'] == 'all' and ((config_params['extra'] == 'dynamic_training_async') or (config_params['extra'] == 'dynamic_training_sync') or (config_params['extra'] == 'dynamic_training_momentumupdate')):
                #    model, optimizer = dynamic_training_utils.dynamic_training(config_params, views_names, model, optimizer, state_before_optim, lr_before_optim, False)

                '''weights_after_backprop = [] # weights after backprop
                for name, param in model.named_parameters():
                    weights_after_backprop.append(param.clone()) # only layer1's weight should update, layer2 is not used
                
                for i in zip(parameter_name, weights_before_backprop, weights_after_backprop):
                    if torch.equal(i[1],i[2]):
                        print(i[0], torch.equal(i[1],i[2]), flush=True)
                '''
                #input('halt')
                

                #performance metrics of training dataset
                correct_train, total_images_train, conf_mat_train, _ = conf_mat_create(pred, train_labels, correct_train, total_images_train, conf_mat_train, config_params['classes'])
                #print('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config_params['maxepochs'], batch_no, batches_train, loss.item()), flush = True)
                #print('Current Time after one batch training:', time.ctime(time.time()), flush=True)

            ''' save the gradients of attention block'''
            #case_size, case_grad_dic_img, case_grad_dic_side = grad_mom_analysis.grad_analysis(model, config_params, views_names, case_size, case_grad_dic_img, case_grad_dic_side, epoch, batch_no)
            
            batch_no=batch_no+1
            optimizer.step() # performs updates using calculated gradients
            if config_params['viewsinclusion'] == 'all' and ((config_params['extra'] == 'dynamic_training_async') or (config_params['extra'] == 'dynamic_training_sync') or (config_params['extra'] == 'dynamic_training_momentumupdate')):
                model, optimizer = dynamic_training(config_params, views_names, model, optimizer, state_before_optim, lr_before_optim, False)
            
            ''' save the momentum of attention block'''
            #case_mom_size, case_mom_dic_img, case_mom_dic_side, avg_mom_img, avg_mom_side = grad_mom_analysis.momentum_analysis(optimizer, config_params, views_names, case_mom_size, case_mom_dic_img, case_mom_dic_side, avg_mom_img, avg_mom_side, epoch, batch_no)

            '''weights_after_backprop = [] # weights after backprop
            for name, param in model.named_parameters():
                weights_after_backprop.append(param.clone()) # only layer1's weight should update, layer2 is not used
            
            for i in zip(parameter_name, weights_before_backprop, weights_after_backprop):
                if torch.equal(i[1],i[2]):
                    print(i[0], torch.equal(i[1],i[2]), flush=True)
            '''

            print('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config_params['maxepochs'], batch_no, batches_train, loss_batch), flush = True)
            print('Current Time after one batch training:', time.ctime(time.time()), flush=True)
        
        if scheduler!=None:
            current_lr=scheduler.get_last_lr()[0]
        else:
            current_lr=optimizer.param_groups[0]['lr']
        print("current lr:",current_lr, flush=True)
        
        '''plt.clf()
        plt.plot(lrs_scheduler)
        plt.savefig(os.path.join(config_params['path_to_output'],'lr_net'+'_'+str(config_params['randseedother'])+'_'+str(config_params['randseeddata'])+'.png'))
        try:
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(config_params['path_to_output'],'lr_class'+'_'+str(config_params['randseedother'])+'_'+str(config_params['randseeddata'])+'.png'))
        except:
            pass
        '''

        running_train_loss = loss_train/total_images_train

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        
        if config_params['usevalidation']:
            correct_test, total_images_val, loss_val, conf_mat_val, auc_val, auc_valmacro = validation(config_params, model, data_iterator_val, batches_val, df_val, epoch)
            valid_loss = loss_val/total_images_val
            results_store_excel(True, True, False, None, correct_train, total_images_train, loss_train, correct_test, total_images_val, loss_val, epoch, conf_mat_train, conf_mat_val, current_lr, auc_val, auc_valmacro, path_to_results_xlsx, path_to_results_text)
        
        if config_params['patienceepochs']:
            modelcheckpoint(valid_loss, model, optimizer, epoch, conf_mat_train, conf_mat_val, running_train_loss, auc_val)
            if modelcheckpoint.early_stop:
                print("Early stopping",epoch+1, flush = True)
                break
        else:
            if config_params['usevalidation']:
                modelcheckpoint(valid_loss, model, optimizer, epoch, conf_mat_train, conf_mat_val, running_train_loss, auc_val)
                test(config_params, model, dataloader_test, batches_test,  df_test, path_to_results_xlsx, 'test_results', epoch)
                #per_model_metrics, conf_mat_test = test(config_params, model, path_to_model, data_iterator_val, batches_val, df_test)
                #evaluation.results_store_excel(True, False, True, per_model_metrics, correct_train, total_images_train, loss_train, None, None, None, epoch, conf_mat_train, None, current_lr, None, path_to_results_xlsx, path_to_results_text)
                #evaluation.write_results_xlsx_confmat(config_params, conf_mat_test, path_to_results_xlsx, 'confmat_train_val_test')
                #evaluation.write_results_xlsx(per_model_metrics, path_to_results_xlsx, 'test_results')
            else:
                save_model(model, optimizer, epoch, running_train_loss, path_to_model)
                per_model_metrics, conf_mat_test = test(config_params, model, path_to_model, data_iterator_val, batches_val, df_test)
                results_store_excel(True, False, True, per_model_metrics, correct_train, total_images_train, loss_train, None, None, None, epoch, conf_mat_train, None, current_lr, None, path_to_results_xlsx, path_to_results_text)
                write_results_xlsx_confmat(config_params, conf_mat_test, path_to_results_xlsx, 'confmat_train_val_test')
                write_results_xlsx(per_model_metrics, path_to_results_xlsx, 'test_results')

        if scheduler!=None: 
            scheduler.step()

        print('Current Time after validation check on the last epoch:', time.ctime(time.time()), flush=True)

    if config_params['usevalidation']:
        write_results_xlsx_confmat(config_params, modelcheckpoint.conf_mat_train_best, path_to_results_xlsx, 'confmat_train_val_test')
        write_results_xlsx_confmat(config_params, modelcheckpoint.conf_mat_test_best, path_to_results_xlsx, 'confmat_train_val_test')

    print('Finished Training')
    
    
def validation(config_params, model, data_iterator_val, batches_val, df_val, epoch):
    """Validation"""
    model.eval()
    total_images=0
    val_loss = 0
    correct = 0
    s=0
    batch_val_no=0
    eval_mode = True
    conf_mat_val=np.zeros((config_params['numclasses'],config_params['numclasses']))

    class_weights_val = class_imbalance(config_params, df_val)

    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss_val, bceloss_val = loss_fn_gmic_initialize(config_params, class_weights_val, test_bool=False)
    else:
        if config_params['activation'] == 'softmax':
            lossfn1 = loss_fn_crossentropy(config_params, class_weights_val, test_bool=False)
        elif config_params['activation'] == 'sigmoid':
            lossfn1 = loss_fn_bce(config_params, class_weights_val, test_bool=False)
    
    with torch.no_grad():   
        for val_idx, val_batch, val_labels, views_names in data_iterator_val:
            val_batch, val_labels = val_batch.to(config_params['device']), val_labels.to(config_params['device'])
            val_labels = val_labels.view(-1)#.float()
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val, _, _, _, _ = model(val_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                    output_patch_val = None
                elif config_params['learningtype'] == 'MIL':
                    output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val, _, _, _, _, output_patch_val = model(val_batch, views_names, eval_mode)
                
                if config_params['activation'] == 'sigmoid':
                    output_batch_local_val = output_batch_local_val.view(-1)
                    output_batch_global_val = output_batch_global_val.view(-1)
                    output_batch_fusion_val = output_batch_fusion_val.view(-1)
                    val_labels = val_labels.float()
                    if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                        val_pred = torch.ge(output_batch_fusion_val, torch.tensor(0.5)).float()
                    else:
                        val_pred = torch.ge(torch.sigmoid(output_batch_fusion_val), torch.tensor(0.5)).float()
                
                elif config_params['activation'] == 'softmax':
                    val_pred = output_batch_fusion_val.argmax(dim=1, keepdim=True)

                loss1 = loss_fn_gmic(config_params, bcelogitloss_val, bceloss_val, output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val, val_labels, class_weights_val, output_patch_val, test_bool=False).item()
                output_val = output_batch_fusion_val
            else:
                if config_params['learningtype'] == 'SIL':
                    output_val = model(val_batch, eval_mode)
                elif config_params['learningtype'] == 'MIL':
                    output_val = model(val_batch, views_names, eval_mode)
                elif config_params['learningtype'] == 'MV':
                    output_val = model(val_batch, views_names, eval_mode)
                if config_params['activation'] == 'sigmoid':
                    if len(output_val.shape)>1:
                        output_val = output_val.squeeze(1)
                    output_val = output_val.view(-1)                                                 
                    val_labels=val_labels.float()
                    if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                        val_pred = torch.ge(output_val, torch.tensor(0.5)).float()
                    else:
                        val_pred = torch.ge(torch.sigmoid(output_val), torch.tensor(0.5)).float()
                    
                    if config_params['classimbalance']=='focalloss':
                        loss1 = sigmoid_focal_loss(output_val, val_labels, alpha=-1, reduction='mean').item()
                    else:
                        if config_params['classimbalance'] == 'poswt':
                            if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                                weight_batch_val = torch.tensor([1, class_weights_val[0]]).to(config_params['device'])[val_labels.long()]
                                lossfn1.weight = weight_batch_val
                        loss1 = lossfn1(output_val, val_labels).item()
                elif config_params['activation'] == 'softmax':
                    val_pred = output_val.argmax(dim=1, keepdim=True)
                    loss1 = lossfn1(output_val, val_labels).item()
            
            if batch_val_no==0:
                val_pred_all = val_pred
                val_labels_all = val_labels
                print(output_val.data.shape, flush=True)
                if config_params['activation'] == 'sigmoid':
                    if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                        output_all_ten = output_val.data
                    else:
                        output_all_ten = torch.sigmoid(output_val.data)
                elif config_params['activation'] == 'softmax':
                    output_all_ten = F.softmax(output_val.data,dim=1)
                    if config_params['numclasses'] < 3:
                        output_all_ten = output_all_ten[:,1]
            else:
                val_pred_all = torch.cat((val_pred_all,val_pred),dim=0)
                val_labels_all = torch.cat((val_labels_all,val_labels),dim=0)
                if config_params['activation'] == 'sigmoid':
                    if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                        output_all_ten = torch.cat((output_all_ten, output_val.data),dim=0)
                    else:
                        output_all_ten = torch.cat((output_all_ten,torch.sigmoid(output_val.data)),dim=0)
                elif config_params['activation'] == 'softmax':
                    if config_params['numclasses'] < 3:
                        output_all_ten = torch.cat((output_all_ten,F.softmax(output_val.data,dim=1)[:,1]),dim=0)
                    else:
                        output_all_ten = torch.cat((output_all_ten,F.softmax(output_val.data,dim=1)),dim=0)

            s = s+val_labels.shape[0]    
            val_loss += val_labels.size()[0]*loss1 # sum up batch loss
            correct, total_images, conf_mat_val, _ = conf_mat_create(val_pred, val_labels, correct, total_images, conf_mat_val, config_params['classes'])
            
            batch_val_no+=1
            print('Val: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config_params['maxepochs'], batch_val_no, batches_val, loss1), flush=True)
    
    print("conf_mat_val:",conf_mat_val, flush=True)
    print("total_images:",total_images, flush=True)
    print("s:",s,flush=True)
    print('\nVal set: total val loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Epoch:{}\n'.format(
        val_loss, val_loss/total_images, correct, total_images,
        100. * correct / total_images,epoch+1), flush=True)
    
    if config_params['numclasses'] > 2:
        auc = metrics.roc_auc_score(val_labels_all.cpu().numpy(), output_all_ten.cpu().numpy(), average='macro', multi_class='ovo')
        auc_wtmacro = metrics.roc_auc_score(val_labels_all.cpu().numpy(), output_all_ten.cpu().numpy(), average='weighted', multi_class='ovo')
    else:
        auc = metrics.roc_auc_score(val_labels_all.cpu().numpy(), output_all_ten.cpu().numpy())
        auc_wtmacro = 0.0

    return correct, total_images, val_loss, conf_mat_val, auc, auc_wtmacro
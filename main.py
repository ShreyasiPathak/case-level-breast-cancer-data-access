# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:43:24 2021

@author: PathakS
"""

import os
import math
import datetime
import argparse
import mlflow

from train_eval.data_loader import dataloader
from train_eval.test import run_test
from train_eval.train import train, model_initialization, set_random_seed

from setup.read_config_file import read_config_file
from setup.read_input_file import input_file_creation
from setup.output_files_setup import output_files

from analysis.attention_wt_extraction import save_attentionwt
from analysis.visualize_roi import run_visualization_pipeline
from analysis.featurevector_hook import save_featurevector, visualize_feature_maps
from analysis.imagelabel_attwt_match import run_imagelabel_attwt_match

if __name__=='__main__':
    #read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        type=str,
        default='',
        help="full path where the config.ini file containing the parameters to run this code is stored",
    )

    parser.add_argument(
        "--num_config_start",
        type=int,
        default=0,
        help="file number of hyperparameter combination to start with; one config file corresponds to one hyperparameter combination",
    )

    parser.add_argument(
        "--num_config_end",
        type=int,
        default=1,
        help="file number of hyperparameter combination to end with; one config file corresponds to one hyperparameter combination",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='train',
        help="model training or test",
    )
    args = parser.parse_args()
    
    #args.config_file_path = '../runs/run1/'
    #args.config_file_path = '/home/runs/run1/'
    args.config_file_path = os.getcwd() + '/runs/run1'
    num_config_start = args.num_config_start
    num_config_end = args.num_config_end

    mode = args.mode

    # Create a new MLflow Experiment
    # set username and password through environment variables. 
    # This is needed for accessing the mlflow client when submitting your code to fe.zgt.nl.
    username = 'your-username' #can be found when logging to fe.zgt.nl
    password = 'your-password' #can be found when logging to fe.zgt.nl
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://localhost:3001")
    
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    
    mlflow.set_experiment(username)

    mlflow.start_run()

    #with mlflow.start_run():
    mlflow.log_params(vars(args))

    #read all instructed config files
    #config_file_names = glob.glob(args.config_file_path+'/config*')
    #config_file_names = sorted(config_file_names, key=lambda x: int(re.search(r'\d+$', x.split('.')[-2]).group()))
    #print("config files to be read:",config_file_names[num_config_start:num_config_end])
    
    #config_file = '/home/runs/run1/config_8.ini'
    config_file = './runs/run1/config_8.ini'
    #for config_file in config_file_names[num_config_start:num_config_end]:
    begin_time = datetime.datetime.now()
    
    config_params = read_config_file(config_file)
    #config_params['batchsize'] = 1
    config_params['path_to_output'] = '/mnt/export/runs/run1' #os.getcwd() + '/runs/run2'
    if not os.path.exists(config_params['path_to_output']):
        os.makedirs(config_params['path_to_output'])
    #config_params['path_to_output'] = "/".join(config_file.split('/')[:-1])
    
    g = set_random_seed(config_params)
    
    if config_params['usevalidation']:
        path_to_model, path_to_results_xlsx, path_to_results_text, path_to_learning_curve, path_to_log_file, path_to_hyperparam_search = output_files(config_params['path_to_output'], config_params, num_config_start, num_config_end)
        df_train, df_val, df_test, batches_train, batches_val, batches_test, view_group_indices_train = input_file_creation(config_params)
        dataloader_train, dataloader_val, dataloader_test = dataloader(config_params, df_train, df_val, df_test, view_group_indices_train, g)
    else:
        path_to_model, path_to_results_xlsx, path_to_results_text, path_to_learning_curve, path_to_log_file = output_files(config_params['path_to_output'], config_params, num_config_start, num_config_end)
        df_train, df_test, batches_train, batches_test, view_group_indices_train = input_file_creation(config_params)
        dataloader_train, dataloader_test = dataloader(config_params, df_train, None, df_test, view_group_indices_train, g)
    
    model, total_params = model_initialization(config_params)

    #df_test.to_csv('./zgt_test_set_case-level_model_randomseed8.csv', sep=';', na_rep='NULL', index=False)
    #input('halt')

    if mode == 'train':
        #training the model
        if config_params['usevalidation']:
            train(config_params, model, path_to_model, dataloader_train, dataloader_val, batches_train, batches_val, df_train, df_val, dataloader_test, batches_test, df_test, path_to_results_xlsx, path_to_results_text)
        else:
            train(config_params, model, path_to_model, dataloader_train, dataloader_test, batches_train, batches_test, df_train, None, None, None, None, None, None)
    
        
    #hyperparameter results
    '''if config_params['usevalidation']:
        config_params['path_to_hyperparam_search'] = path_to_hyperparam_search
        config_params['config_file'] = config_file.split('/')[-1]
        run_test(config_params, model, path_to_model, dataloader_val, batches_val, df_val, path_to_results_xlsx, 'hyperparam_results')
        #per_model_metrics_val, _ = run_test(config_params, model, path_to_model, dataloader_val, batches_val, df_val, path_to_results_xlsx)
        ##hyperparam_details = [config_file.split('/')[-1], config_params['lr'], config_params['wtdecay'], config_params['sm_reg_param'], config_params['trainingmethod'], config_params['optimizer'], config_params['patienceepochs'], config_params['batchsize']] + per_model_metrics_val
        #evaluation.write_results_xlsx(hyperparam_details, path_to_hyperparam_search, 'hyperparam_results')
    '''
    
    #test the model
    #print(df_test['Views'].str.split('+').str.len().groupby())
    run_test(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_results_xlsx, 'test_results', 'test')
    
    #save attention weights
    #path_to_attentionwt = "/".join(config_file.split('/')[:-1]) #"C:/Users/PathakS/OneDrive - Universiteit Twente/PhD/projects/radiology breast cancer/breast-cancer-multiview-mammogram-codes/multiinstance results/results/ijcai23/error_analysis_plots/"
    #print(path_to_attentionwt)
    #save_attentionwt(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_attentionwt)

    #visualize saliency maps and ROI candidates
    #run_visualization_pipeline(config_params, model, path_to_model, dataloader_test, df_test)

    #match image labels to attention weights
    #run_imagelabel_attwt_match(config_params, model, path_to_model, dataloader_test, df_test)

    #save feature vector 
    '''path_to_featurevector = "/".join(config_file.split('/')[:-1]) #"C:/Users/PathakS/OneDrive - Universiteit Twente/PhD/projects/radiology breast cancer/breast-cancer-multiview-mammogram-codes/multiinstance results/results/ijcai23/error_analysis_plots/"
    print(path_to_featurevector)
    #save_featurevector(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_featurevector)
    visualize_feature_maps(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_featurevector)
    '''
    
    mlflow.end_run()

    f = open(path_to_log_file,'a')
    f.write("Model parameters:"+str(total_params/math.pow(10,6))+'\n')
    f.write("Start time:"+str(begin_time)+'\n')
    f.write("End time:"+str(datetime.datetime.now())+'\n')
    f.write("Execution time:"+str(datetime.datetime.now() - begin_time)+'\n')
    f.close()
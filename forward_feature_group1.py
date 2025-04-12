#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch

import numpy as np
from utils import load_model, forward_pass
from get_dataloader import prepare_data, get_data


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract feature for supervised models.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-s', '--source', type=str, default='imagenet',
                        help='source dataset for pre-trained models (e.g., imagenet)')
    parser.add_argument('-b', '--batch-size', type=int, default=256, 
                        help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, 
                        help='the size of the input images')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='CUDA or CPU training (cuda | cpu)')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    
    args = parser.parse_args()
    args.norm = not args.no_norm
    pprint(args)

    # load dataset
    dset, data_dir, num_classes, metric = get_data(args.dataset)
    args.num_classes = num_classes
    
    train_loader, val_loader, trainval_loader, test_loader, all_loader = prepare_data(
        dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm)
    
    print(f'Train:{len(train_loader.dataset)}, Val:{len(val_loader.dataset)},' 
            f'TrainVal:{len(trainval_loader.dataset)}, Test:{len(test_loader.dataset)} '
            f'AllData:{len(all_loader.dataset)}')
    
    trainval_loader = all_loader

    fpath = os.path.join('./results_f', args.source, 'group1')
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    models_by_params = {
        # Group 1: 7 models with lowest parameter count
        1: ['mobilenet_v2', 'mnasnet1_0', 'googlenet', 'inception_v3', 'resnet34', 'densenet121', 'densenet169'],
        # Group 2-4: Intermediate parameter counts
        2: ['mobilenet_v2', 'mnasnet1_0', 'googlenet', 'resnet34', 'densenet121', 'resnet50', 'densenet169'],
        3: ['googlenet', 'inception_v3', 'resnet34', 'densenet121', 'resnet50', 'densenet169', 'densenet201'],
        4: ['inception_v3', 'resnet50', 'densenet169', 'densenet201', 'resnet101', 'resnet152', 'densenet201'],
        # Group 5: 7 models with highest parameter count
        5: ['densenet121', 'densenet169', 'densenet201', 'resnet50', 'resnet101', 'resnet152', 'inception_v3']
    }
    
    if args.model != 'deepcluster-v2':
        models_hub = [args.model]
    else:
        models_hub = ['inception_v3', 'mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201', 
                 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
    
    for model in models_hub:
        args.model = model
        args.checkpoint_path = os.path.join('./models/group1/checkpoints', args.source)
        
        model_npy_feature = os.path.join(fpath, f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join(fpath, f'{args.model}_{args.dataset}_label.npy')

        if os.path.exists(model_npy_feature) and os.path.exists(model_npy_label):
            print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
            continue
            
        model, fc_layer = load_model(args)
        X_trainval_feature, X_output, y_trainval = forward_pass(trainval_loader, model, fc_layer)   
        
        #X_trainval_feature, y_trainval = forward_pass_feature(trainval_loader, model)   
        if args.dataset == 'voc2007':
            y_trainval = torch.argmax(y_trainval, dim=1)
        print(f'x_trainval shape:{X_trainval_feature.shape} and y_trainval shape:{y_trainval.shape}')
        
        np.save(model_npy_feature, X_trainval_feature.numpy())
        np.save(model_npy_label, y_trainval.numpy())
        print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint
from rank_correlation import (load_score, recall_k, rel_k, pearson_coef, 
                            wpearson_coef, w_kendall_metric, kendall_metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate transferability metrics.')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', 
                        help='name of the target dataset to evaluate on')
    parser.add_argument('-s', '--source', type=str, default='imagenet',
                        help='source dataset for pre-trained models (e.g., imagenet)')
    parser.add_argument('-f', '--finetune-type', type=str, default='head-training',
                        help='ground-truth type for transferability estimation (head-training, full-training)')
    parser.add_argument('-c', '--complexity', type=int, default=1, choices=range(1, 6),
                        help='complexity degree of model-hub (1=lowest params, 5=highest params)')
    parser.add_argument('-me', '--method', type=str, default='logme', 
                        help='name of used transferability metric')
    args = parser.parse_args()
    pprint(args)

    # Full fine-tuning accuracy
    full_training_acc = {
        'aircraft': {'resnet34': 84.06, 'resnet50': 84.64, 'resnet101': 85.53, 'resnet152': 86.29, 'densenet121': 84.66, 
                    'densenet169': 84.19, 'densenet201': 85.38, 'mnasnet1_0': 66.48, 'mobilenet_v2': 79.68, 
                    'googlenet': 80.32, 'inception_v3': 80.15}, 
        'caltech101': {'resnet34': 91.15, 'resnet50': 91.98, 'resnet101': 92.38, 'resnet152': 93.1, 'densenet121': 91.5, 
                    'densenet169': 92.51, 'densenet201': 93.14, 'mnasnet1_0': 89.34, 'mobilenet_v2': 88.64, 
                    'googlenet': 90.85, 'inception_v3': 92.75}, 
        'cars': {'resnet34': 88.63, 'resnet50': 89.09, 'resnet101': 89.47, 'resnet152': 89.88, 'densenet121': 89.34, 
                    'densenet169': 89.02, 'densenet201': 89.44, 'mnasnet1_0': 72.58, 'mobilenet_v2': 86.44, 
                    'googlenet': 87.76, 'inception_v3': 87.74}, 
        'cifar10': {'resnet34': 96.12, 'resnet50': 96.28, 'resnet101': 97.39, 'resnet152': 97.53, 'densenet121': 96.45, 
                    'densenet169': 96.77, 'densenet201': 97.02, 'mnasnet1_0': 92.59, 'mobilenet_v2': 94.74, 
                    'googlenet': 95.54, 'inception_v3': 96.18}, 
        'cifar100': {'resnet34': 81.94, 'resnet50': 82.8, 'resnet101': 84.88, 'resnet152': 85.66, 'densenet121': 82.75, 
                    'densenet169': 84.26, 'densenet201': 84.88, 'mnasnet1_0': 72.04, 'mobilenet_v2': 78.11, 
                    'googlenet': 79.84, 'inception_v3': 81.49}, 
        'dtd': {'resnet34': 72.96, 'resnet50': 74.72, 'resnet101': 74.8, 'resnet152': 76.44, 'densenet121': 74.18, 
                    'densenet169': 74.72, 'densenet201': 76.04, 'mnasnet1_0': 70.12, 'mobilenet_v2': 71.72, 
                    'googlenet': 72.53, 'inception_v3': 72.85}, 
        'flowers': {'resnet34': 95.2, 'resnet50': 96.26, 'resnet101': 96.53, 'resnet152': 96.86, 'densenet121': 97.02, 
                    'densenet169': 97.32, 'densenet201': 97.1, 'mnasnet1_0': 95.39, 'mobilenet_v2': 96.2, 
                    'googlenet': 95.76, 'inception_v3': 95.73},
        'food': {'resnet34': 81.99, 'resnet50': 84.45, 'resnet101': 85.58, 'resnet152': 86.28, 'densenet121': 84.99, 
                    'densenet169': 85.84, 'densenet201': 86.71, 'mnasnet1_0': 71.35, 'mobilenet_v2': 81.12, 
                    'googlenet': 79.3, 'inception_v3': 81.76}, 
        'pets': {'resnet34': 93.5, 'resnet50': 93.88, 'resnet101': 93.92, 'resnet152': 94.42, 'densenet121': 93.07, 
                    'densenet169': 93.62, 'densenet201': 94.03, 'mnasnet1_0': 91.08, 'mobilenet_v2': 91.28, 
                    'googlenet': 91.38, 'inception_v3': 92.14},
        'sun397': {'resnet34': 61.02, 'resnet50': 63.54, 'resnet101': 63.76, 'resnet152': 64.82, 'densenet121': 63.26, 
                    'densenet169': 64.1, 'densenet201': 64.57, 'mnasnet1_0': 56.56, 'mobilenet_v2': 60.29, 
                    'googlenet': 59.89, 'inception_v3': 59.98}, 
        'voc2007': {'resnet34': 84.6, 'resnet50': 85.8, 'resnet101': 85.68, 'resnet152': 86.32, 'densenet121': 85.28, 
                    'densenet169': 85.77, 'densenet201': 85.67, 'mnasnet1_0': 81.06, 'mobilenet_v2': 82.8, 
                    'googlenet': 82.58, 'inception_v3': 83.84}
    }
    
    # Head-only fine-tuning accuracy
    head_training_acc = {'aircraft': {'inception_v3': 28.21, 'mobilenet_v2': 42.24, 'mnasnet1_0': 41.72, 'densenet121': 43.61, 'densenet169': 47.15, 'densenet201': 46.39, 'resnet34': 38.19, 'resnet50': 40.63, 'resnet101': 41.21, 'resnet152': 42.98, 'googlenet': 36.22},
    'caltech101': {'inception_v3': 88.48, 'mobilenet_v2': 87.35, 'mnasnet1_0': 87.85, 'densenet121': 90.03, 'densenet169': 90.76, 'densenet201': 91.31, 'resnet34': 89.8, 'resnet50': 89.75, 'resnet101': 89.81, 'resnet152': 91.42, 'googlenet': 88.31},
    'cars': {'inception_v3': 27.6, 'mobilenet_v2': 49.77, 'mnasnet1_0': 46.19, 'densenet121': 51.78, 'densenet169': 56.2, 'densenet201': 57.32, 'resnet34': 32.04, 'resnet50': 50.91, 'resnet101': 50.6, 'resnet152': 52.07, 'googlenet': 43.83},
    'cifar10': {'inception_v3': 69.87, 'mobilenet_v2': 76.97, 'mnasnet1_0': 69.55, 'densenet121': 81.39, 'densenet169': 83.08, 'densenet201': 84.52, 'resnet34': 78.61, 'resnet50': 83.57, 'resnet101': 85.24, 'resnet152': 85.33, 'googlenet': 78.45},
    'cifar100': {'inception_v3': 46.39, 'mobilenet_v2': 57.46, 'mnasnet1_0': 37.49, 'densenet121': 62.11, 'densenet169': 64.53, 'densenet201': 67.51, 'resnet34': 59.43, 'resnet50': 65.41, 'resnet101': 67.64, 'resnet152': 67.81, 'googlenet': 59.73},
    'dtd': {'inception_v3': 61.28, 'mobilenet_v2': 67.77, 'mnasnet1_0': 65.69, 'densenet121': 68.09, 'densenet169': 69.95, 'densenet201': 70.64, 'resnet34': 66.7, 'resnet50': 70.74, 'resnet101': 69.57, 'resnet152': 70.74, 'googlenet': 66.12},
    'flowers': {'inception_v3': 83.01, 'mobilenet_v2': 92.27, 'mnasnet1_0': 92.37, 'densenet121': 93.23, 'densenet169': 94.15, 'densenet201': 93.01, 'resnet34': 90.71, 'resnet50': 93.05, 'resnet101': 92.3, 'resnet152': 93.06, 'googlenet': 89.53},
    'food': {'inception_v3': 46.31, 'mobilenet_v2': 62.6, 'mnasnet1_0': 62.65, 'densenet121': 65.37, 'densenet169': 67.81, 'densenet201': 68.11, 'resnet34': 60.56, 'resnet50': 65.79, 'resnet101': 66.5, 'resnet152': 67.55, 'googlenet': 55.34},
    'pets': {'inception_v3': 85.85, 'mobilenet_v2': 89.73, 'mnasnet1_0': 89.56, 'densenet121': 91.46, 'densenet169': 92.6, 'densenet201': 92.57, 'resnet34': 91.27, 'resnet50': 91.76, 'resnet101': 92.34, 'resnet152': 92.67, 'googlenet': 89.41},
    'sun397': {'inception_v3': 63.72, 'mobilenet_v2': 73.25, 'mnasnet1_0': 79.63, 'densenet121': 76.37, 'densenet169': 80.78, 'densenet201': 80.38, 'resnet34': 71.96, 'resnet50': 83.29, 'resnet101': 75.61, 'resnet152': 75.72, 'googlenet': 76.82},
    'voc2007': {'inception_v3': 77.01, 'mobilenet_v2': 80.88, 'mnasnet1_0': 81.18, 'densenet121': 82.73, 'densenet169': 84.07, 'densenet201': 83.34, 'resnet34': 82.46, 'resnet50': 83.28, 'resnet101': 83.85, 'resnet152': 84.13, 'googlenet': 80.32},}
    
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
    
    selected_models = models_by_params[args.complexity]
    
    # Choose the correct ground truth based on finetune type
    if args.finetune_type == 'head-training':
        finetune_acc = head_training_acc
    else:  # full-training is the default fallback
        finetune_acc = full_training_acc
    
    filtered_finetune_acc = {}
    
    for dataset in finetune_acc:
        filtered_finetune_acc[dataset] = {model: acc for model, acc in finetune_acc[dataset].items() 
                                        if model in selected_models}
    
    dset = args.dataset
    metric = args.method
    finetune_type = args.finetune_type
    
    score_path = f'./results_metrics/{args.source}/group1/{metric}/{dset}_metrics.json'
    
    if not os.path.exists(score_path):
        print(f"Score file not found: {score_path}")
        print(f"Please run evaluate_metric_group1_cpu.py with -s {args.source} -me {metric} -d {dset} first")
        exit(1)
    
    score, _ = load_score(score_path)
    
    filtered_score = {model: score[model] for model in selected_models if model in score}
    
    if len(filtered_score) < len(selected_models):
        missing_models = [model for model in selected_models if model not in score]
        print(f"Warning: Missing scores for models: {missing_models}")
        print(f"Please run evaluate_metric_group1_cpu.py for these models")
    
    tw_score = w_kendall_metric(filtered_score, filtered_finetune_acc, dset)
    
    print(f"Using {finetune_type} as ground truth with complexity level {args.complexity}")
    print(f"Selected models: {selected_models}")
    print(f"Kendall correlation - dataset: {dset:12s} {metric}: {tw_score:2.3f}")
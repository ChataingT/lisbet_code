import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import json
import random
import sys
sys.path.append(r"/home/share/schaer2/thibaut/humanlisbet/lisbet_code/utils")
from utils.models import OptimizedMLP, CNN, LSTMModel
from utils.model_optim import trainer
import argparse
def process_data_with_windows_mlp(df, window_size=200, stride=90):
    """
    Converts the input dictionary into multiple windows for MLP training.
    
    Args:
        data_dict (dict): Dictionary with video data.
            - Keys: Video IDs
            - Values: {"keypoints": np.array of shape (num_frames, 34), "diag": binary}
        window_size (int): Number of frames per window.
        stride (int): Number of frames to slide for the next window.
        
    Returns:
        X (np.array): Flattened input features for MLP of shape (num_windows, 34 * window_size)
        y (np.array): Binary labels of shape (num_windows,)
    """
    idx_vid, X, y = [], [], []
    

    for video_id, video_data in df.groupby('video'):
        diag = video_data.iloc[0].diagnosis
        vd = video_data.drop(['video', 'diagnosis'], axis=1).to_numpy()
        num_frames= len(vd)
        # Create windows
        for start in range(0, num_frames - window_size + 1, stride):
            window = vd[start : start + window_size]
            X.append(window.flatten())
            y.append(diag)
            idx_vid.append(video_id)

    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return idx_vid, X, y


def process_data_with_windows_cnn(df, window_size=200, stride=90):
    """
    Converts the input dictionary into multiple windows for MLP training.
    
    Args:
        data_dict (dict): Dictionary with video data.
            - Keys: Video IDs
            - Values: {"keypoints": np.array of shape (num_frames, 34), "diag": binary}
        window_size (int): Number of frames per window.
        stride (int): Number of frames to slide for the next window.
        
    Returns:
        X (np.array): Flattened input features for MLP of shape (num_windows, 34 * window_size)
        y (np.array): Binary labels of shape (num_windows,)
    """
    idx_vid, X, y = [], [], []
    
    for video_id, video_data in df.groupby('video'):
        diag = video_data.iloc[0].diagnosis
        vd = video_data.drop(['video', 'diagnosis'], axis=1).to_numpy()
        num_frames= len(vd)
        # Create windows
        for start in range(0, num_frames - window_size + 1, stride):
            window = vd[start : start + window_size]
            X.append(window)
            y.append(diag)
            idx_vid.append(video_id)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return idx_vid, X, y

def process_data_with_windows_lstm(df, window_size=200, stride=90):
    """
    Converts the input dictionary into multiple windows for MLP training.
    
    Args:
        data_dict (dict): Dictionary with video data.
            - Keys: Video IDs
            - Values: {"keypoints": np.array of shape (num_frames, 34), "diag": binary}
        window_size (int): Number of frames per window.
        stride (int): Number of frames to slide for the next window.
        
    Returns:
        X (np.array): Flattened input features for MLP of shape (num_windows, 34 * window_size)
        y (np.array): Binary labels of shape (num_windows,)
    """
    idx_vid, X, y = [], [], []
    
    for video_id, video_data in df.groupby('video'):
        diag = video_data.iloc[0].diagnosis
        vd = video_data.drop(['video', 'diagnosis'], axis=1).to_numpy()
        num_frames= len(vd)
        
        # Create windows
        for start in range(0, num_frames - window_size + 1, stride):
            window = vd[start : start + window_size]
            X.append(window)
            y.append(diag)
            idx_vid.append(video_id)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return idx_vid, X, y


def main(args=None):
    parser = argparse.ArgumentParser(description='Benchmark 2')
    parser.add_argument('--seed', type=int, help='seed')

    args = parser.parse_args(args)
    # mlp

    rooroot = r"/home/share/schaer2/thibaut/humanlisbet/output"
    # root_fit = r"/home/share/schaer2/thibaut/humanlisbet/bet_fits"
    mapping_path = r"/home/share/schaer2/thibaut/humanlisbet/datasets/humans/category_mapping.json"
    label_path = r"/home/share/schaer2/thibaut/humanlisbet/datasets/humans/humans_annoted.label.json"


    folders = os.listdir(rooroot)

    # seeds = [random.randint(100, 10009) for i in range(5)]
    # seeds = [21,54,68,74,82]
    # seeds = [139,360,4148,7630,8522]
    # seeds = [42,66,1789]
    seeds = [int(args.seed)]

    model='mlp'
    for fol in folders:
        if fol.endswith('ipynb'):
            continue
        print(fol)
        fol = "lisbet128x1-14258188-14"
        for seed in seeds:
            out = os.path.join(rooroot, fol,'pred', f"out_{model}_{seed}")
            os.makedirs(out, exist_ok=True)
            datapath = os.path.join(rooroot, fol,"embedding_train.npy")
            dataval = os.path.join(rooroot, fol, "embedding_test.npy")

            bf = os.path.join(rooroot, fol,"models",fol, 'model_config.yml')
            with open(bf, 'r') as fd:
                params = yaml.safe_load(fd)
            os.makedirs(out, exist_ok=True)
            print(params)


            # Hyperparameters
            HIDDEN_SIZE = 128
            OUTPUT_SIZE = 1  # Binary classification
            LEARNING_RATE = 1e-5
            EPOCHS = 500
            BATCH_SIZE = 64
            # seed = seed
            test_ratio = 0.8
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            DROPOUT_RATE = 0.3
            verbose = False
            EMB_DIM = params['emb_dim']
            INPUT_SIZE = params['emb_dim'] * 200 # nbr kypoints * windows size in data preprocessing 

            # Parameter dictionary
            run_parameters = {
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "output_size": OUTPUT_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": seed,
                "test_ratio": test_ratio,
                "dropout_rate": DROPOUT_RATE,
                "verbose": verbose,
                "emb_dim":EMB_DIM,
                "data":'emb',
                'model':model,
                'run_id': fol
            }
            run_parameters.update(params['out_dim'])

            with open(os.path.join(out, 'parameters.json'), 'w') as fd:
                json.dump(run_parameters, fd, indent=4)


            # Initialize the model
            model = OptimizedMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

            dfm = trainer(out, run_parameters, mapping_path, label_path, datapath, device, dataval, model, process_data_with_windows_mlp)

            del model
            del dfm

    # CNN


    model = 'cnn'
    for fol in folders:
        if fol.endswith('ipynb'):
            continue
        print(fol)
        for seed in seeds:
            out = os.path.join(rooroot, fol,'pred', f"out_{model}_{seed}")
            os.makedirs(out, exist_ok=True)
            datapath = os.path.join(rooroot, fol,"embedding_train.npy")
            dataval = os.path.join(rooroot, fol, "embedding_test.npy")

            bf = os.path.join(rooroot, fol,"models",fol, 'model_config.yml')
            with open(bf, 'r') as fd:
                params = yaml.safe_load(fd)
            os.makedirs(out, exist_ok=True)
            print(params)

            # Hyperparameters
            INPUT_SIZE = params['emb_dim']
            HIDDEN_SIZE = 128
            WINDOW = 200
            OUTPUT_SIZE = 1  # Binary classification
            LEARNING_RATE = 1e-5
            EPOCHS = 500
            BATCH_SIZE = 64
            # seed = seed
            test_ratio = 0.8
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            DROPOUT = 0.3
            verbose = False
            NUM_FILTER=64
            KERNEL_SIZE=30   # look at one second
            POOL_SIZE=2
            EMB_DIM = INPUT_SIZE

            task = {k: 1 for k in params['task'].split(',')}
            # Parameter dictionary
            run_parameters = {
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "output_size": OUTPUT_SIZE,
                "window": WINDOW,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": seed,
                "test_ratio": test_ratio,
                "dropout": DROPOUT,
                "num_filter": NUM_FILTER,
                "verbose": verbose,
                "kernel_size":KERNEL_SIZE,
                'pool_size':POOL_SIZE,
                "emb_dim":EMB_DIM,
                "data":'emb',
                'model':model,
                'run_id': fol
            }
            run_parameters.update(task)


            with open(os.path.join(out, 'parameters.json'), 'w') as fd:
                json.dump(run_parameters, fd, indent=4)

            # Initialize the model, loss, and optimizer
            model = CNN(input_size=INPUT_SIZE, sequence_length=WINDOW, num_filters=NUM_FILTER, 
                        kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)

            dfm = trainer(out, run_parameters, mapping_path, label_path, datapath, device, dataval, model, process_data_with_windows_cnn)



    # LSTM

    model='lstm'
    for fol in folders:
        if fol.endswith('ipynb'):
            continue
        print(fol)
        
        for seed in seeds:

            out = os.path.join(rooroot, fol,'pred', f"out_{model}_{seed}")
            os.makedirs(out, exist_ok=True)
            datapath = os.path.join(rooroot, fol,"embedding_train.npy")
            dataval = os.path.join(rooroot, fol, "embedding_test.npy")

            bf = os.path.join(rooroot, fol,"models",fol, 'model_config.yml')
            with open(bf, 'r') as fd:
                params = yaml.safe_load(fd)
            os.makedirs(out, exist_ok=True)
            print(params)


            # Hyperparameters
            INPUT_SIZE = params['emb_dim']
            HIDDEN_SIZE = 64
            WINDOW = 200
            OUTPUT_SIZE = 1  # Binary classification
            LEARNING_RATE = 1e-5
            EPOCHS = 500
            BATCH_SIZE = 64
            seed = seed
            test_ratio = 0.8
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            DROPOUT = 0.3
            verbose = False
            NUM_LAYER=1
            EMB_DIM = INPUT_SIZE

            task = {k: 1 for k in params['task'].split(',')}

            # Parameter dictionary
            run_parameters = {
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "output_size": OUTPUT_SIZE,
                "window": WINDOW,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": seed,
                "test_ratio": test_ratio,
                "dropout": DROPOUT,
                "num_layer": NUM_LAYER,
                "verbose": verbose,
                "emb_dim":EMB_DIM,
                "data":'emb',
                'model':model,
                'run_id': fol
            }
            run_parameters.update(task)

            with open(os.path.join(out, 'parameters.json'), 'w') as fd:
                json.dump(run_parameters, fd, indent=4)

            

            # Initialize the model, loss, and optimizer
            model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYER, DROPOUT).to(device)

            dfm = trainer(out, run_parameters, mapping_path, label_path, datapath, device, dataval, model, process_data_with_windows_lstm)

            del model 
            del dfm

if __name__ == '__main__':
    main()
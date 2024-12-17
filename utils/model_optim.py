
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
sys.path.append(r"C:\Users\chataint\Documents\projet\humanlisbet\baseline")
from utils import load_embedding, AutismDataset, debug_metrics, get_metrics, compute_validation, load_h5_data

# early stopping based of f1

class EarlyStoppingMetric:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth', warm_up=5):
        """
        Args:
            patience (int): How long to wait after the last improvement.
            verbose (bool): If True, prints a message when validation improves.
            delta (float): Minimum change to qualify as improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_losss_min = 0  # Initialize with 0
        self.warm_up = warm_up
        self.best_epoch = 0


    def __call__(self, epoch, metric, model):
        """Return if has found best"""
        if self.warm_up > epoch:
            return False
        
        score = metric  # Negative because we want to minimize loss

        if self.best_score is None or score > self.best_score + self.delta:
            # Improvement detected
            self.best_score = score
            self.best_epoch = epoch

            self.save_checkpoint(metric, model)
            self.counter = 0
            return True
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                tqdm.write(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def save_checkpoint(self, test_losss, model):
        """Save the best model"""
        if self.verbose:
            tqdm.write(f"Validation loss improved ({self.test_losss_min:.6f} --> {test_losss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.test_losss_min = test_losss

class EarlyStoppingLoss:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth', warm_up=5):
        """
        Args:
            patience (int): How long to wait after the last improvement.
            verbose (bool): If True, prints a message when validation improves.
            delta (float): Minimum change to qualify as improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_losss_min = 0  # Initialize with 0
        self.warm_up = warm_up
        self.best_epoch = 0

    def __call__(self, epoch, test_loss, model):
        if self.warm_up > epoch:
            return
        
        score = test_loss  # Negative because we want to minimize loss

        if self.best_score is None or score > self.best_score + self.delta:
            # Improvement detected
            self.best_score = test_loss
            self.best_epoch = epoch
            self.save_checkpoint(test_loss, model)
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                tqdm.write(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, test_losss, model):
        """Save the best model"""
        if self.verbose:
            tqdm.write(f"Validation loss improved ({self.test_losss_min:.6f} --> {test_losss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.test_losss_min = test_losss


def trainer(out, run_parameters, mapping_path, label_path, datapath, device, dataval, model, process_data_with_windows):

    with open(mapping_path, 'r') as fd:
        mapping = json.load(fd)

    with open(label_path, 'r') as fd:
        dc = pd.DataFrame(json.load(fd))
    dc.video = dc.video.astype(dtype=pd.Int32Dtype())
    dc.diagnosis = dc.diagnosis.map({value: key for key, value in mapping.items()})
    dc.diagnosis = dc.diagnosis.astype(dtype=pd.Int16Dtype())
    mapping = {int(key): value for key, value in mapping.items()}

    if datapath.endswith('.h5'):
        # records, labels = load_h5_data(datapath)
        df = load_h5_data(datapath)
    else:
        df = load_embedding(datapath, dc, emb_dim=run_parameters['emb_dim'])

    rec_train, rec_test = train_test_split(
                df, test_size=run_parameters['test_ratio'], random_state=run_parameters['seed'], stratify=df.diagnosis
            )


    # with open(os.path.join(out, "rec_test"), 'wb') as fd:
    #     pickle.dump(rec_test, fd)
    # with open(os.path.join(out, "rec_train"), 'wb') as fd:
    #     pickle.dump(rec_train, fd)

    idx_vid_test, X_test, y_test = process_data_with_windows(rec_test)
    idx_vid_train, X_train, y_train = process_data_with_windows(rec_train)


    dataset = AutismDataset(X_train, y_train, idx_vid_train, device=device)
    train_loader = DataLoader(dataset, batch_size=run_parameters['batch_size'], shuffle=True)

    dataset_test = AutismDataset(X_test, y_test, idx_vid_test, device=device)
    test_loader = DataLoader(dataset_test, batch_size=run_parameters['batch_size'], shuffle=True)


    pos_weight = torch.tensor(((y_train.squeeze() -1).sum() *-1) / y_train.squeeze().sum())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=run_parameters['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    early_stopping = EarlyStoppingMetric(patience=20, verbose=run_parameters['verbose'], path=os.path.join(out, 'best_model.pth'), warm_up=1)

    dfm = pd.DataFrame()
    # Training loop
    for epoch in tqdm(range(run_parameters['epochs']), desc="Training Progress", unit="epoch"):
        metrics = {'epoch':epoch}
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_X, batch_y, _) in tqdm(enumerate(train_loader), 
                                                total=len(train_loader), 
                                                desc=f"Training {epoch + 1}", 
                                                leave=False, disable=not(run_parameters['verbose'])):
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Adjust learning rate
        scheduler.step(epoch_loss)

        metrics['loss']=epoch_loss / len(train_loader)
        # Validation phase
        model.eval()  # Set model to evaluation mode
        test_targets, test_predictions, videos, test_loss = [], [], [], 0
        with torch.no_grad():
            for batch_idx, (batch_X, batch_y, idx_video) in tqdm(enumerate(test_loader), 
                                                total=len(test_loader), 
                                                desc=f"Validation {epoch + 1}", 
                                                leave=False, disable=not(run_parameters['verbose'])):
                
                # Forward pass
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y.squeeze())
                test_loss += loss
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                # Store predictions and targets
                test_predictions.extend(torch.round(outputs).cpu().numpy())  # Convert logits to binary predictions
                test_targets.extend(batch_y.cpu().numpy())
                videos.extend(idx_video.cpu().numpy())
        
        test_loss = test_loss.cpu().numpy() / len(test_loader)
        # Compute validation metrics
        val_accuracy = accuracy_score(test_targets, test_predictions)
        val_precision = precision_score(test_targets, test_predictions, zero_division=0)
        val_recall = recall_score(test_targets, test_predictions, zero_division=0)
        val_f1 = f1_score(test_targets, test_predictions, zero_division=0)

        if run_parameters['verbose']:
            tqdm.write(f"Epoch {epoch + 1}/{run_parameters['epochs']}, Loss: {epoch_loss / len(train_loader):.4f} Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            
        if True:
            debug_metrics(test_targets, test_predictions, videos, mapping, epoch, out)
        metrics['acc'] = val_accuracy
        metrics['prec'] = val_precision
        metrics['rec'] = val_recall
        metrics['f1'] = val_f1
        metrics['test_loss'] = test_loss

        dfm = pd.concat([dfm, pd.DataFrame(metrics, index=[0])], ignore_index=True)

        # Check early stopping
        found_best = early_stopping(metric=val_f1, model=model, epoch=epoch)
        if found_best:
            f1_best_fr, f1_best_vid = get_metrics(test_targets, test_predictions, videos, mapping, out)
            run_parameters['fr-test-f1'] = f1_best_fr
            run_parameters['vid-test-f1'] = f1_best_vid
        # early_stopping(test_loss=test_loss, model=model, epoch=epoch)
        if early_stopping.early_stop:
            tqdm.write(f"Early stopping triggered. Training terminated. Best model at {early_stopping.best_epoch} with score={early_stopping.best_score}")
            break

    print(f"Epoch {epoch + 1}/{run_parameters['epochs']}, Loss: {epoch_loss / len(train_loader):.4f} Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(out,"last_model.pth"))
    dfm.to_csv(os.path.join(out, 'metrics.csv'))



    
    if datapath.endswith('.h5'):
        df = load_h5_data(dataval)
    else:
        df = load_embedding(dataval, dc, emb_dim=run_parameters['emb_dim'])

    idx_vid_val, X_val, y_val = process_data_with_windows(df)

    db_val = AutismDataset(X_val, y_val,idx_vid_val, device=device)
    val_loader = DataLoader(db_val, batch_size=run_parameters['batch_size'])

    y_true, y_pred, videos = [], [],[]
    with torch.no_grad():
        for batch_idx, (batch_X, batch_y, idx_video) in tqdm(enumerate(val_loader), 
                                                total=len(val_loader), 
                                                desc=f"Validation", 
                                                leave=False, disable=not(run_parameters['verbose'])):
            y_pred.extend(model(batch_X).cpu().numpy().squeeze().round())
            y_true.extend(batch_y.cpu().numpy().squeeze())
            videos.extend(idx_video.cpu().numpy())

    f1_best_fr, f1_best_vid = compute_validation(y_true, y_pred, videos, out, mapping)
    
    run_parameters['fr-val-f1'] = f1_best_fr
    run_parameters['vid-val-f1'] = f1_best_vid

    
    with open(os.path.join(out, f"info_{run_parameters['seed']}.json"), 'w') as fd:
        json.dump(run_parameters, fd, indent=4)

    return dfm
    
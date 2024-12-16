import torch
from torch.utils.data import  Dataset
from collections import defaultdict
import h5py
from pathlib import Path
import numpy as np
import pandas as pd

def load_h5_data(datapath):
    records = defaultdict(dict)

    def load_dataset(name, node):
        if isinstance(node, h5py.Dataset):
            record_name = str(Path(name).parent)
            dataset_name = str(Path(name).name)
            records[record_name][dataset_name] = node[()]
            for key, val in node.attrs.items():
                records[record_name][key] = val

    with h5py.File(datapath, "r") as h5file:
        h5file.visititems(load_dataset)

    labels = []
    for val in records.values():
        labels.append(val['diag'][0])
    records = list(tuple(records.items()))

    dr = pd.DataFrame()
    for vid in records:
        id = vid[0]
        frs = vid[1]['keypoints']
        diag = vid[1]['diag'][0]
        df = pd.DataFrame(frs)
        df['video']=int(id)
        df['diagnosis']=int(diag)
        dr = pd.concat([dr, df],ignore_index=True)

    # return records, labels
    df.video = df.video.astype(np.int64)
    df.diagnosis = df.diagnosis.astype(dtype=pd.Int16Dtype())
    return dr


# Sample dataset implementation
class AutismDataset(Dataset):
    def __init__(self, X, y,idx_video=None, device='cpu'):
        """
        X: np.array of shape (num_samples, num_features)
        y: np.array of shape (num_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        self.idx_video = idx_video

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.idx_video[idx]
    


def load_embedding(datapath, dc, emb_dim):
    emb_train = np.load(datapath, allow_pickle=True)

    # Initialize an empty list to collect rows for the DataFrame
    rows = []

    # Iterate over the array to flatten the structure
    for video, coords in emb_train:
        for idx, coord in enumerate(coords):
            rows.append([video, idx] + coord.tolist())

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=['video', 'frame'] + [f'em_{i}' for i in range(emb_dim)])
    df.video = df.video.astype(np.int64)

    df = df.drop(columns='frame')
    df = pd.merge(left=df, right=dc, left_on='video', right_on='video', how='left')
    # dict_classes = {v:k for k,v in enumerate(df['diagnosis'].unique())}
    # df.diagnosis = df.diagnosis.map(mapping)

    return df
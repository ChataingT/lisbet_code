import os
import pandas as pd
import argparse
import tqdm
import numpy as np

def main(args=None):
    parser = argparse.ArgumentParser(description='Benchmark 2')
    parser.add_argument('--input', type=str, help='Input directory')

    args = parser.parse_args(args)
    input_dir = args.input
    for list_xp in tqdm.tqdm(os.listdir(input_dir)):
        if list_xp.endswith('13'):
            continue
        if list_xp.endswith('11') or list_xp.endswith('12'):

            df_tr = pd.DataFrame()
            # list_xp = "lisbet128x1-14258188-14"
            list_xp_path_train = os.path.join(input_dir, list_xp, 'TRAIN')
            for r,d,f in tqdm.tqdm(os.walk(list_xp_path_train), leave=False):
                for file in f:
                    if '.csv' in file:
                        df = pd.read_csv(os.path.join(r, file), header=0, index_col=0)
                        df['video'] = os.path.basename(r)
                        df_tr = pd.concat([df_tr, df], axis=0)
            df_tr = df_tr.to_numpy()
            np.save(os.path.join(input_dir,list_xp,"embedding_train.npy"), df_tr)

            df_te = pd.DataFrame()
            list_xp_path_test = os.path.join(input_dir, list_xp, 'TEST')
            for r,d,f in tqdm.tqdm(os.walk(list_xp_path_test), leave=False):
                for file in f:
                    if '.csv' in file:
                        df = pd.read_csv(os.path.join(r, file), header=0, index_col=0)
                        df['video'] = os.path.basename(r)
                        df_te = pd.concat([df_te, df], axis=0)

            df_te = df_te.to_numpy()
            np.save(os.path.join(input_dir,list_xp,"embedding_test.npy"), df_te)
        # break
    return

if __name__ == "__main__":
    main()
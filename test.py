from  lisbet_dev.src.lisbet.console import parse_cl_bet, parse_cl_unsup
from  lisbet_dev.tools.export_embedder import main as export_embedder
from  lisbet_dev.tools.export_embedding import main as export_embedding
import os

if __name__ == '__main__':
    root = r"/tf"
    run_id = "42"
    epoch = "75"
    data_path=r"/tf/datasets/humans/humans_50-50_annoted.h5"
    window_size="200"
    
    args = ["train",
         #   "--epochs","2",
           "--epochs",epoch,
          #  "--task","nwp,vsp,dmp",
           "--task","nwp,vsp,dmp,bcf",
           # "--task","bcf",
           # "--train_sample","8",
           "--train_sample","8192",
           "--dataset","GenericH5",
           "--datapath",data_path,
           "--run_id",run_id,
           "--emb_dim","32",
          #  "--num_layers","4",
           "--num_layers","1",
          #  "--num_heads","4",
           "--num_heads","1",
           "--hidden_dim","2048",
           "--seed","20230312",
           "--dev_ratio","0.2",
           "--feature_name_bcf","diag",
           "--batch_size", "64",
           "--window_size", window_size,
        #    "--output_path", os.path.join(root, "test"),
         #   "--save_checkpoints",
           "--save_weights",
           # "--verbose",
            # "--debug",
           "--jit_compile"

    ]
    
    # parse_cl_bet(args)
    args = [
        "--root", root ,
        "--run_id", run_id,
        "--epoch", epoch,
        "--skeletons_keypoints", "34"
    ]
    
    # export_embedder(args)

    args = [
        "--root", root ,
        "--run_id", run_id,
        "--data_path", data_path,
        "--window_size", window_size
    ]
    
    # export_embedding(args)

    args = ["segment",
           "--output_path", os.path.join(root, r"hmm_fits"),
           "--model_path", os.path.join(root, rf"bet_embedders/{run_id}/model_config.json"),
           "--weights_path", os.path.join(root, rf"bet_embedders/{run_id}/model_weights.hdf5"),
           "--dataset","GenericH5",
           "--datapath", data_path,
           "--num_iter", "300",
           "--num_state", "40",
           "--window_size", "200"
           
    ]
    
    parse_cl_unsup(args)

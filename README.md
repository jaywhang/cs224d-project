# cs224d-project

## Usage examples

### Character Model (plain text)
    python2 -O trainer.py --data_type=text --data_path=input.txt --config=lstm --model_type='char' --output_dir=results/test

### Character Model (Penn Treebank)
    python2 -O trainer.py --data_type=ptb --data_path=ptb --config=lstm --model_type='char' --output_dir=results/test

### Word Model (Penn Treebank)

    python2 -O trainer.py --data_type=ptb --data_path=ptb --config=lstm --model_type='word' --output_dir=results/test

### Hyperparameter exploration (overriding default hyperparameters)
    python2 -O trainer.py --data_type=ptb --data_path=ptb --model_type='char' --output_dir=results --me=50 --bs=64 --sl=100 --hs=500 --kp=1.0 --ct=lstm

### Restarting training (Specify --restart_training, loads params from checkpoint under filepath)
    python2 -O trainer.py --restart_training --me=100 \
      --data_type=ptb --data_path=ptb --model_type='char' --output_dir=results --bs=64 --sl=100 --hs=500 --kp=1.0 --ct=lstm


## `trainer_refactor` branch
The `ef` parameter specifies every how many iterations we eval valid loss.
This branch doesn't support restarting and TensorBoard stuff for now.

    python2 -O trainer.py \
        --data_type=ptb --data_path=ptb --model_type='char' --output_dir=dir \
        --me=2 --bs=256 --sl=10 --hs=100 --kp=1.0 --ct=lstm --ef=100



## Other utils

### Comparing loss values from different models
Inside `results` folder, run:

    ./plot_graphs.py \
        -a \             # Needed if you want to use folder names as labels
        bs_64_ct_bnlstm_hs_300_kp_1.0_lr_0.002_mgn_1.0_op_adam_sl_100_vs_50 \
        bs_64_ct_lstm_hs_300_kp_1.0_lr_0.002_mgn_1.0_op_adam_sl_100_vs_50

Specifying labels:
    ./plot_graphs.py \
        path/to/folder1 label_for_folder1 \
        path/to/folder2 label_for_folder2 \
        path/to/folder3 label_for_folder3

The output files are named `valid_losses.pdf` and `valid_pps.pdf`.


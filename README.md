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

### Restarting training (Specify --restart_training, --start_epoch, --me)
    python2 -O trainer.py --restart_training --start_epoch=51 --me=100 \
      --data_type=ptb --data_path=ptb --model_type='char' --output_dir=results --bs=64 --sl=100 --hs=500 --kp=1.0 --ct=lstm


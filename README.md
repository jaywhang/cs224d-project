# cs224d-project

## Usage

* Character Model (plain text)
    python2 -O trainer.py --data_type=text --data_path=input.txt --config=lstm --model_type='char' --output_dir=results/test

* Character Model (Penn Treebank)
    python2 -O trainer.py --data_type=ptb --data_path=ptb --config=lstm --model_type='char' --output_dir=results/test

* Word Model (Penn Treebank)
    python2 -O trainer.py --data_type=ptb --data_path=ptb --config=lstm --model_type='word' --output_dir=results/test

* Hyperparameter exploration (overriding default hyperparameters)
    python2 -O trainer.py --data_type=text --data_path=char_rnn/input.txt --config=lstm --model_type='char' --output_dir=delete/me --me=2 --bs=256 --sl=10 --hs=50 --kp=1.0


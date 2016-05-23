# cs224d-project

## Usage

* Character Model (plain text)
    python2 -O trainer.py --data_type=text --data_path=input.txt --config=lstm --model_type='char' --plot_dir=text_char

* Character Model (Penn Treebank)
    python2 -O trainer.py --data_type=ptb --data_path=ptb --config=lstm --model_type='char' --plot_dir=ptb_char

* Word Model (Penn Treebank)
    python2 -O trainer.py --data_type=ptb --data_path=ptb --config=lstm --model_type='word' --plot_dir=ptb_word


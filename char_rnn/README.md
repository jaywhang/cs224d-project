Character-level stacked LSTM model with dropout.

Expects `input.txt` as training data.

Note that this treats the newline character `\n` just like any other token,
so sampled sentences often have random linebreaks.

Usage:
    python2 run_small_model.py

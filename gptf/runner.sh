#!/bin/bash          

#python3 train.py --lr=0.05 --batch_size=20 --n_inputs=50 --n_epoch=10 
# Default method reaches 0.93 r2-score

# Now we use inputs on a grid
python3 train.py --lr=0.1 --batch_size=20 --n_inputs=7 --n_epoch=50 
# Default method reaches 0.91 r2-score

#!/bin/bash          

#python3 train.py --lr=0.05 --batch_size=20 --n_inputs=50 --n_epoch=10 
# Default method reaches 0.93 r2-score

# Now we use inputs on a grid
# python3 train.py --lr=0.05 --batch_size=20 --n_inputs=7 --n_epoch=50 
# This method reaches 0.93 r2-score

# I fixed the noise factor at a low value, it is now untrainable
# Also, now interpolation is used to compute K_nm
#python3 train.py --lr=0.05 --batch_size=20 --n_inputs=10 --n_epoch=50 
# This method reaches approx 0.88 r2-score

# Now I set variational parameters to be tt / kron.
#python3 train.py --lr=0.005 --batch_size=20 --n_inputs=10 --n_epoch=500
# This method reaches approx 0.83 r2-score

# More Iterations
# python3 train.py --lr=0.005 --batch_size=20 --n_inputs=10 --n_epoch=1000 --refresh_stats=True
# This method reaches approx 0.94 r2-score

# Initialization for mu and Sigma
python3 train.py --lr=0.05 --batch_size=20 --n_inputs=10 --n_epoch=500 --refresh_stats=True
# This method reaches approx 0.94 r2-score

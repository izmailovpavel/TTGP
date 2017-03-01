# Initialization for mu and Sigma
#python3 train.py --lr=.08 --batch_size=100 --n_inputs=10 --n_epoch=500 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False
python3 train.py --lr=.005 --batch_size=100 --n_inputs=10 --n_epoch=1000 \
                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True

DATASETS_DIR_1="/Users/IzmailovPavel/Documents/Education/Programming/DataSets/"
DATASETS_DIR_2="/Users/IzmailovPavel/Documents/Education/Projects/GPtf/data/"

# Synthetic Data (300, 2)
# Initial Values [0.3, 0.8, 0.3]
DATA_DIR=$DATASETS_DIR_2"synthetic(300,2)/"
python3 train.py --lr=.1 --batch_size=100 --n_inputs=7 --n_epoch=15 \
                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
#python3 train.py --lr=.1 --batch_size=100 --n_inputs=7 --n_epoch=50 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
# Achieves 0.92.

# Synthetic Data (1000, 3)
# Initial Values [0.7, 0.2, 0.2]
#DATA_DIR=$DATASETS_DIR_2"synthetic(1000,3)/"
#python3 train.py --lr=.1 --batch_size=100 --n_inputs=7 --n_epoch=5 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
#python3 train.py --lr=.01 --batch_size=100 --n_inputs=7 --n_epoch=5 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
# Achieves 0.97. The second part of the training isn't really required.

# Synthetic Data (3000, 3). 
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_2"synthetic_hard(3000,3)/"
#python3 train.py --lr=.01 --batch_size=100 --n_inputs=15 --n_epoch=30 \
#                 --refresh_stats=True --mu_ranks=7 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
#python3 train.py --lr=.005 --batch_size=100 --n_inputs=15 --n_epoch=30 \
#                 --refresh_stats=True --mu_ranks=7 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
# Achieves 0.91

# Synthetic Data (10000, 4). Harder dataset, it is generated from a fast varying
# GP in 3D space. Normal methods don't really work on it.
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_2"synthetic4d(10000,4)/"
#python3 train.py --lr=.05 --batch_size=500 --n_inputs=15 --n_epoch=30 \
#                 --refresh_stats=True --mu_ranks=15 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
#python3 train.py --lr=.005 --batch_size=500 --n_inputs=15 --n_epoch=30 \
#                 --refresh_stats=True --mu_ranks=15 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
# Achieves 0.62. There is a room for improvement.

# mg data (1385, 6). This dataset is easy for standard methods.
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_1"Regression/mg(1385, 6).txt"
#python3 train.py --lr=.05 --batch_size=100 --n_inputs=10 --n_epoch=30 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True
#python3 train.py --lr=.01 --batch_size=100 --n_inputs=10 --n_epoch=20 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True
# Achieves 0.7. tt-ranks 3 -> 0.68. tt-ranks 5 -> 0.7.

# abalone (4177, 8).
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_1"Regression/abalone(4177, 8).txt"
#python3 train.py --lr=0.005 --batch_size=100 --n_inputs=10 --n_epoch=5 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True
#python3 train.py --lr=.0005 --batch_size=100 --n_inputs=10 --n_epoch=15 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True
# Achieves 0.52.

# cadata (4177, 8).
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_1"Regression/cadata(20640, 8).txt"
#python3 train.py --lr=0.005 --batch_size=500 --n_inputs=10 --n_epoch=5 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True
#python3 train.py --lr=.005 --batch_size=500 --n_inputs=10 --n_epoch=15 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True
# Achieves 0.72.

# bodyfat (252, 14).
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_1"Regression/bodyfat(252, 14).txt"
#python3 train.py --lr=0.005 --batch_size=50 --n_inputs=10 --n_epoch=100 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True
#python3 train.py --lr=.00005 --batch_size=50 --n_inputs=10 --n_epoch=100 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True
# Achieves 0.72.

# Combined Cycle Power Plant Data (9568, 4).
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_2"CCPP(9568,4)/"
#python3 train.py --lr=.05 --batch_size=100 --n_inputs=15 --n_epoch=10 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
#python3 train.py --lr=.005 --batch_size=100 --n_inputs=15 --n_epoch=10 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
# Achieves .93 

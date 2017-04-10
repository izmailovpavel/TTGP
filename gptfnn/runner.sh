DATASETS_DIR_1="/Users/IzmailovPavel/Documents/Education/Programming/DataSets/"
DATASETS_DIR_2="/Users/IzmailovPavel/Documents/Education/Projects/GPtf/data/"

export CUDA_VISIBLE_DEVICES=""

#DATA_DIR=$DATASETS_DIR_2"simple_projection(300,3)/"
#python3 train.py --lr=.01 --batch_size=100 --n_inputs=10 --n_epoch=20 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True \
#                 --d=2
#python3 train.py --lr=.01 --batch_size=50 --n_inputs=10 --n_epoch=200 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True \
#                 --d=2
# Achieves 0.91 if P starts from PCA. 0.84 if starts from eye.

#DATA_DIR=$DATASETS_DIR_2"projection_redundant(500,3)/"
#python3 train.py --lr=.01 --batch_size=100 --n_inputs=10 --n_epoch=20 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True
#                 --d=2
#python3 train.py --lr=.01 --batch_size=50 --n_inputs=10 --n_epoch=200 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True \
#                 --d=2
# Achieves 0.86 starting from eye.

#DATA_DIR=$DATASETS_DIR_2"projection_hard(3000,8[3])/"
#python3 train.py --lr=.01 --batch_size=300 --n_inputs=25 --n_epoch=30 \
#                 --refresh_stats=True --mu_ranks=20 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True \
#                 --d=3
#python3 train.py --lr=.005 --batch_size=50 --n_inputs=25 --n_epoch=60 \
#                 --refresh_stats=True --mu_ranks=20 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy" --stoch=True \
#                 --d=3
# Achieves 0.77 starting from eye.

# mg data (1385, 6). This dataset is easy for standard methods.
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_1"Regression/mg(1385, 6).txt"
#python3 train.py --lr=.05 --batch_size=100 --n_inputs=10 --n_epoch=100 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True \
#                 --d=4
#python3 train.py --lr=.005 --batch_size=100 --n_inputs=10 --n_epoch=100 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True \
#                 --d=4

# bodyfat (252, 14).
# Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_1"Regression/bodyfat(252, 14).txt"
#python3 train.py --lr=0.0005 --batch_size=50 --n_inputs=10 --n_epoch=100 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=False \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True \
#                 --d=3
#python3 train.py --lr=.00005 --batch_size=50 --n_inputs=10 --n_epoch=200 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=True \
#                 --datadir="$DATA_DIR" --datatype="svmlight" --stoch=True \
#                 --d=3
# Achieves 0.82


# digits (1797, 64).
# Initial Values [0.7, 0.2, 0.1]
DATA_DIR=$DATASETS_DIR_2"digits(1797,64)/"
#python3 train.py --lr=0.01 --batch_size=100 --n_inputs=15 --n_epoch=15 \
#                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=False \
#                 --datadir="$DATA_DIR" --datatype="numpy" --stoch=True \
#                 --d=3
python3 train.py --lr=.001 --batch_size=100 --n_inputs=15 --n_epoch=50 \
                 --refresh_stats=True --mu_ranks=10 --load_mu_sigma=True \
                 --datadir="$DATA_DIR" --datatype="numpy" --stoch=True \
                 --d=3
# Achieves 0.7 for d=3.

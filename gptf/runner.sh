DATASETS_DIR_1="/Users/IzmailovPavel/Documents/Education/Programming/DataSets/"
DATASETS_DIR_2="/Users/IzmailovPavel/Documents/Education/Projects/GPtf/data/"

#Synthetic Data (300, 2)
#Initial Values [0.7, 0.2, 0.1]
#DATA_DIR=$DATASETS_DIR_2"synthetic(300,2)/"
#python3 train.py --lr=.1 --batch_size=100 --n_inputs=7 --n_epoch=25 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy"
#python3 train.py --lr=.05 --batch_size=100 --n_inputs=7 --n_epoch=100 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
#                 --datadir=$DATA_DIR --datatype="numpy"
# Achieves 0.93. The second part of the training isn't really required.

#Synthetic Data (1000, 3)
#Initial Values [0.7, 0.2, 0.2]
DATA_DIR=$DATASETS_DIR_2"synthetic(1000,3)/"
#python3 train.py --lr=.1 --batch_size=100 --n_inputs=7 --n_epoch=25 \
#                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=False \
#                 --datadir=$DATA_DIR --datatype="numpy"
python3 train.py --lr=.01 --batch_size=100 --n_inputs=7 --n_epoch=100 \
                 --refresh_stats=True --mu_ranks=5 --load_mu_sigma=True \
                 --datadir=$DATA_DIR --datatype="numpy"
# Achieves 0.93. The second part of the training isn't really required.

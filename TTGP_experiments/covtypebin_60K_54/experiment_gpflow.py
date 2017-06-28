import numpy as np
import numpy.random as rnd
import time
import GPflow
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Logger
st = time.time()
epoch_start = time.time()
logt = []
logx = []
logf = []
def logger(x):
    if (logger.i % iter_per_epoch) == 0:
        logx.append(x)
        logf.append(m._objective(x)[0])
        logt.append(time.time() - st)
        print('Epoch:', logger.i / iter_per_epoch)
        if (len(logt) > 1):
            print('Epoch took:', logt[-1] - logt[-2])
        print('\t',m._objective(x)[0])
    logger.i+=1
logger.i = 1

# Data loading
x_tr = np.load('data/x_tr.npy')
x_te = np.load('data/x_te.npy')
y_te = np.load('data/y_te.npy')[:, None]
y_tr = np.load('data/y_tr.npy')[:, None]
scaler_x = StandardScaler()
x_tr = scaler_x.fit_transform(x_tr)
x_te = scaler_x.transform(x_te)

# Inducing Inputs
M = 1000
kmeans = KMeans(n_clusters=M, n_init=3, max_iter=10, random_state=241, verbose=1)
kmeans.fit(x_tr)
Z = kmeans.cluster_centers_
#data = np.copy(x_tr) 
#np.random.seed(417)
#np.random.shuffle(data)
#Z = data[:M, :]

# Kernel and model
batch_size = 500
kern = GPflow.kernels.RBF(54, variance=0.7, lengthscales=0.3)
m = GPflow.svgp.SVGP(x_tr, y_tr, kern, Z=Z, minibatch_size=batch_size,
                likelihood=GPflow.likelihoods.Bernoulli())

# Batch sizes
m.X.minibatch_size = batch_size
m.Y.minibatch_size = batch_size

# Training
n_epoch = 10
iter_per_epoch = int(y_tr.size / batch_size)
print(iter_per_epoch)
maxiter = iter_per_epoch * n_epoch
m.Z.fixed = True
m.optimize(method=tf.train.AdamOptimizer(learning_rate=1e-2), maxiter=maxiter, 
        callback=logger)

pred = (m.predict_y(x_te)[0] > 0.5).astype(int)
print()
print('Final accuracy:', accuracy_score(y_te.astype(int), pred.astype(int)))
print()
m.optimize(method=tf.train.AdamOptimizer(learning_rate=1e-3), maxiter=maxiter, 
        callback=logger)
pred = (m.predict_y(x_te)[0] > 0.5).astype(int)
print()
print('Final accuracy:', accuracy_score(y_te.astype(int), pred.astype(int)))
print()
m.optimize(method=tf.train.AdamOptimizer(learning_rate=1e-4), maxiter=maxiter, 
        callback=logger)
pred = (m.predict_y(x_te)[0] > 0.5).astype(int)
print()
print('Final accuracy:', accuracy_score(y_te.astype(int), pred.astype(int)))
print()

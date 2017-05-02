# CIFAR-10 dataset

CIFAR-10 dataset. Here we experiment with a convolutional embedding. 

- A 1728-conv64-conv64-dense64-dense4 embedding gives 0.79.
- A 1728-conv64-conv64-dense384-dense4 embedding gives 0.82.
- A 1728-conv64-conv64-dense512-128-dense4 embedding gives 0.845.
- A 1728-conv64-conv64-pool-conv128-conv128-pool-conv128-conv128-dense1536-dense512-dense5 embedding gives 0.9.

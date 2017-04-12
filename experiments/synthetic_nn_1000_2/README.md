# synthetic_nn dataset

Synthetic data — a two-dimensional process, generated from a GP with Neural 
Network kernel.

First experiments show that it's rather hard to train the model, at least for 
this dataset. Now I first train the method with lr=0.1 for 50 epochs and then 
train it again with lr=0.01 for 50 more epochs. And currently I use a very simple
NN.

![Data](https://cloud.githubusercontent.com/assets/14368801/24977992/b7eea4c0-1fd7-11e7-8415-c0cd0d34a25f.png)
- Linear Embedding to dim d=2 achieves 0.86 r2 score.

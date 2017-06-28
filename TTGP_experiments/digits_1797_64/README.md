# DIGITS dataset

Digits dataset. Here we experiment with linear and NN embedding. 

## Regression

Here we train an GP on DIGITS with regression likelihood.

![LinEmbedding](https://cloud.githubusercontent.com/assets/14368801/24927164/b6765736-1f06-11e7-9a25-5dc2f3ec8e11.png)

- Linear Embedding to dim d=2 achieves 0.62 r2 score.
- Linear Embedding to dim d=4 achieves 0.7 r2 score.

![NNEmbedding](https://cloud.githubusercontent.com/assets/14368801/24982997/ffc95c08-1fec-11e7-9f7d-856628291913.png)

- NN Embedding to dim d=2 with (64->100->100->2) fully-connected architecture 
  achieves 0.8 r2 score.

## Classification

Here we use a multiclass GP with NN embedding. 
- NN Embedding to dim d=2 with (64->20->20->2) fully-connected architecture 
  achieves 0.87 accuracy. 
- NN Embedding to dim d=2 with (64->20->20->4) fully-connected architecture 
  achieves 0.9 accuracy. 
![NNEmbeddingClass](https://cloud.githubusercontent.com/assets/14368801/25273973/cecfce9c-2696-11e7-8281-f4de84919e8d.png)

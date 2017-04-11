# projection_redundant dataset

Synthetic data. Here I generated the data from a two dimensional GP, then added
a noise feature and rotated the resulting data in the 3D space. 

The method is able to find the subspace, that determines the process, not taking
the noisy feature into account.

![Data](https://cloud.githubusercontent.com/assets/14368801/24928460/32586764-1f0b-11e7-9774-f0a540c44c87.png)
- Linear Embedding to dim d=2 achieves 0.9 r2 score.

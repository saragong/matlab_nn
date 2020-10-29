# Code Sample: Neural Network Implementation in MATLAB

This MATLAB code, part of an assignment for a machine learning class, implements a two-layer neural network (without regularization) that performs multiclass classification for 20 x 20 images of handwritten digits.

The number of iterations and the number of nodes in the hidden layer can be adjusted in "run.m". This script also outputs the empirical risk (using a 0-1 loss function) of the trained classifier. I have also written code in "nnCostFunction.m" to perform backpropagation. Note that there are two prewritten functions that accompanied the assignment: "displayData.m", which displays the image data, and "fmincg.m", which is a nonlinear programming solver.
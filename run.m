%% Code Sample: Neural Network Implementation in MATLAB
% Implements a 2-layer neural network to recognize handwritten digits.

% initialize
clear ; close all; clc
addpath('data', 'scripts', 'scripts/prewritten')
load('data.mat');         % load design matrix X and vector of labels y

m = size(X, 1);           % number of training examples
input_layer_size  = 400;  % 20x20 input images of digits
hidden_layer_size = 20;   % 20 nodes in hidden layer
num_labels = 10;          % 10 labels, from 1 to 10   

% select 100 random data points to display (using prewritten function displayData)
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

% randomly initialize weights for each of the two layers
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)]; % unroll

% set number of iterations
options = optimset('MaxIter', 100);

% define the cost function as a function of weight matrices only, holding
% other parameters fixed
func = @(a) nnCostFunction(a, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y);

% perform gradient descent using prewritten function "fmincg"
[nn_params, cost] = fmincg(func, initial_nn_params, options);

% obtain model parameters
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% predict fitted values
pred = predict(Theta1, Theta2, X);

% calculate the empirical risk (0-1 loss function) and print to console
risk = mean(double(pred ~= y));
fprintf('\nAccuracy on Training Sample: %f\n', 1 - risk);

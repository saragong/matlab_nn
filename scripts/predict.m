%% predict.m
% function to output predicted labels of a matrix X given a 2-layer neural
% network with parameters Theta1, Theta2

function p = predict(Theta1, Theta2, X)

m = size(X, 1);                 % training sample size
num_labels = size(Theta2, 1);   % number of labels

p = zeros(size(X, 1), 1);
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

end

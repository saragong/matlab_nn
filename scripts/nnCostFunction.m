%% nnCostFunction.m
% implements the cost function for a 2-layer neural network by computing
% cost and gradient

function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y)

% obtain model parameters                               
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);                      % number of training examples (5000)
J = 0;                               % cost                  

% initialize gradient vectors
Theta1_grad = zeros(size(Theta1));      
Theta2_grad = zeros(size(Theta2));

% define matrices
a1 = [ones(m, 1) X];                 % m x length of feature vector
z2 = a1*Theta1';                     % m x number nodes in hidden layer
a2 = sigmoid(z2);                    % m x number nodes in hidden layer

a2 = [ones(size(a2, 1), 1) a2];      % m x number nodes in hidden layer + 1
z3 = a2*Theta2';                     % m x 10
a3 = sigmoid(z3);                    % m x 10

h = a3;                              % output 

Y = zeros(size(y, 1), size(a3, 2));  % convert to binary classification problem
for i = 1:size(Y, 1)
    Y(i,y(i)) = 1;
end

% compute logistic loss of neural network (with respect to Y)
for i = 1:size(Y, 1)                 
    for k = 1:size(Y, 2)
        J = J + (Y(i,k).*log(h(i,k)) + (1 - Y(i,k)).*log(1 - h(i,k)));
    end
end    

% obtain the unregularized cost
J = (-1/m)*J;

% perform backpropagation
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));    

for i = 1:m
    a1i = a1(i, :)';                % length of feature vector (plus bias) x 1       
    z2i = Theta1*a1i;               % number of nodes in hidden layer x 1
    a2i = sigmoid(z2i);             % number of nodes in hidden layer x 1

    a2i = [1; a2i];                 % number of nodes in hidden layer (plus bias) x 1
    z3i = Theta2*a2i;               % 10 x 1     
    a3i = sigmoid(z3i);             % 10 x 1
    
    yi = Y(i, :)';                  % 10 x 1
    delta3i = a3i - yi;             % compute deltas ("error terms")
    delta2i = (Theta2'*delta3i).*a2i.*(1-a2i); 
    
    D2 = D2 + delta3i*(a2i');    	% accumulate gradients
    D1 = D1 + delta2i(2:end)*(a1i');   
    
end

% obtain the unregularized gradients
Theta1_grad = (1/m)*D1;             
Theta2_grad = (1/m)*D2; 
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
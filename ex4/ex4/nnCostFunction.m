function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 


X = [ones(m,1) X]; %include row of 1's

%%%%%%%%forward propag
z1 = sigmoid(Theta1 * X');
a2 = [ones(1, size(z1, 2)); z1];
a3 = sigmoid(Theta2 * a2);
h = a3;

% Note that this time we do not transpose a3 to create h as to make
% the following matrix multiplication slightly simpler




%%%%%%%%COST

%y into a matrix instead of vector, 10x5000 matrix instead of vector 5000x1
yMatrix = zeros(num_labels, m);
for i=1:num_labels,
    yMatrix(i,:) = (y==i);
endfor


%yk is given by the matrix 
J = (sum( sum( -1*yMatrix.*log(h) - (1 - yMatrix).*log(1-h) ) ))/m;

% First, we toss the first columns of each Theta(i) matrix.

Theta1Reg = Theta1(:,2:size(Theta1,2));

Theta2Reg = Theta2(:,2:size(Theta2,2));

Reg = (lambda/(2*m)) * (sum(sum( Theta1Reg.^2 )) + sum( sum( Theta2Reg.^2 ) ));

J = J + Reg;

% -------------------------------------------------------------

% Implement Part II -- implementing back propagation
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively.

for k = 1:m,
   

    %forward prop
    a1 = X(k,:);
    z2 = Theta1 * a1';

    a2 = sigmoid(z2);
    a2 = [1 ; a2];
    z3 =  Theta2 * a2;

    
    a3 = sigmoid(z3);   %final layer

    %backwards prop 
    d3 = a3 - yMatrix(:,k);
    
    % need to add bias node
    z2 = [1 ; z2];
    d2 = (Theta2' * d3) .* sigmoidGradient(z2);
    
    % need to remove bias node
    d2 = d2(2:end);

    Theta2_grad = (Theta2_grad + d3 * a2');
    Theta1_grad = (Theta1_grad + d2 * a1);

endfor;

% Now divide everything (element-wise) by m to return the partial
% derivatives. Note that for regularization these will have to
% removed/commented out.

% Theta2_grad = Theta2_grad ./ m;
% Theta1_grad = Theta1_grad ./ m;

% -------------------------------------------------------------

% Implement Part III -- Regularization with cost function/gradients
%
% Part 3: Implement regularization with the cost function and gradients.
      
%Delta(l(i,j)) = 1/m*delta(l(i,j)) + lambda/m*(Theta(l(i,j))
% for j >= 1

%  when l = 0
Theta1_grad(:,1) = Theta1_grad(:,1)./m;
Theta2_grad(:,1) = Theta2_grad(:,1)./m;

% l > 0
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)./m + ( (lambda/m) * Theta1(:,2:end) );
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)./m + ( (lambda/m) * Theta2(:,2:end) );


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
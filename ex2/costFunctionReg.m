function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));



h = sigmoid(X*theta);

reg_term = lambda/(2*m) * (theta(2:end, :)' * theta(2:end,:));
J = sum(-y .* log(h) - (1-y) .* log(1-h)) * (1/m) + reg_term;


for i = 1 : size(theta, 1)
    grad(i) = (1 / m) * sum( (h - y) .* X(:, i) );
end


end




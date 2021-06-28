function [J, grad] = costFunction(theta, X, y)


% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);

J = -(1 / m) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) );

for i = 1 : size(theta, 1)
    grad(i) = (1 / m) * sum( (h - y) .* X(:, i) );
end


end

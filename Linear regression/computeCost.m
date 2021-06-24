function J = computeCost(X, y, theta)
m = length(y); % number of training examples
J = 0;

hypothesis = X*theta;
sqrerror = (hypothesis - y).^2;
J = 1/(2*m)* sum(sqrerror);

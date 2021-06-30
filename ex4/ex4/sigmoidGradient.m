function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z


g = zeros(size(z));



%the derivative/gradient can be written as
g = sigmoid(z).*(1-sigmoid(z));






end

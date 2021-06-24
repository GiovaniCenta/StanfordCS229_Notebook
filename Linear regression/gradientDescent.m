function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
   

 
    
    x = X(:,2);
    hypothesis = theta(1) + (theta(2)*x);

    theta(1) = theta(1) - alpha * (1/m) * sum(hypothesis-y);   %theta0
    theta(2)  = theta(2) - alpha * (1/m) * sum((hypothesis - y) .* x); %theta1

    theta = [theta(1); theta(2)];
       
    J_history(iter) = computeCost(X, y, theta);

end

end
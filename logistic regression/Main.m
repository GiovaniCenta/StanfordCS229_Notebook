%% Initialization
clear; close all; clc

%%%%% Load Data
%  The first two columns contains the variables and the third column
%  contains the label.

%data = load('Data.txt');
%X = data(:, [1, 2]); y = data(:, 3);
data = csvread('nba_logreg.csv');
X = data(:,[3:4]); y = data(:, 21);


%%%%% Plot the data
plotData(X,y);
legend('true','false');
xlabel('variable1');
ylabel('variable2');
%Helps understand the problem


%%%%% cost and gradient

% m is the number of samples
% n is the number of features
[m,n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];


% initial theta values are set to zero
% the correct values will be determined later by the optmization algorithm
initial_theta = zeros(n + 1, 1);    %initialize all values with zero

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

%%%%%%%Optimizing using gradient descent

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
  
 % Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Variable 1')
ylabel('Variable 2')

% Specified in plot order
legend('True', 'False')
hold off;

var1 = 15;
var2 = 5;
prob = sigmoid([1 var1 var2] * theta);
fprintf(['For variable1 =  %f and  variable 2 = %f, the model predicted a ' ...
         'probability of %f \n\n'], var1,var2,prob);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

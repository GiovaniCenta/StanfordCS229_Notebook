
%% Clear and Close Figures
clear ; close all; clc

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset without normalization: \n');
fprintf(' x = [%.2f %.2f], y = %.2f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('First 10 examples from the dataset with normalization: \n');
fprintf(' x = [%.2f %.2f], y = %.2f \n', [X(1:10,:) y(1:10,:)]');

%%%%% Gradient Descent

% Parameters
alpha = 0.12;
num_iters = 50;

theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
set(gcf, 'Name', 'Convergence cost graph');

plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price
sqrfeet = 2000;
bedrooms = 4;

price = [1,(sqrfeet - mu(1))/sigma(1),(bedrooms - mu(2))/sigma(2)]*theta; 

fprintf(['Predicted price of a %.2f sq-ft, %d bedrooms house ' ...
         '(using gradient descent):\n $%.2f\n'], sqrfeet,bedrooms,price);
         
         
% Plotting Training and regressioned data. 
%Source:https://github.com/everpeace/ml-class-assignments/blob/master/ex1.Linear_Regression/mlclass-ex1/ex1_multi.m

X = [ones(m, 1) data(:, 1:2)]; %denormalize features
figure;
plot3(X(:,2),X(:,3),y,"ko");
xlabel('sq-ft');
ylabel('#bedroom');
zlabel('price');
grid;
hold on;
xx = linspace(0,5000,25);
yy = linspace(1,5,25);
zz = zeros(size(xx,2),size(yy,2));
for i=1:size(xx,2)
for j=1:size(yy,2)
  zz(i,j) = [1 (xx(i)-mu(1))/sigma(1) (yy(j)-mu(2))/sigma(2)]*theta;
end
end
mesh(xx,yy,zz);
title('Result of Gradient Descent');
legend('Training data', 'Linear regression');
        
         
         
         
         
         
         
         
         
         
         
%%%%% Normal equations         
         
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %.2f \n', theta);

% Estimate the price 
price =  [1 sqrfeet bedrooms]*theta;
fprintf(['Predicted price of a %.2f sq-ft, %d bedrooms house ' ...
         '(using normal equations):\n $%.2f\n'], sqrfeet, bedrooms,price);
         
         
% Plotting Training and regressioned data.
%Source:https://github.com/everpeace/ml-class-assignments/blob/master/ex1.Linear_Regression/mlclass-ex1/ex1_multi.m
fprintf('Plotting Training and regressioned results by solving normal equation.\n');
figure;
plot3(X(:,2),X(:,3),y,"ko");
xlabel('sq-ft');
ylabel('#bedroom');
zlabel('price');
grid;
hold on;
xx = linspace(0,5000,25);
yy = linspace(1,5,25);
zz = zeros(size(xx,2),size(yy,2));
for i=1:size(xx,2)
for j=1:size(yy,2)
  zz(i,j) = [1 xx(i) yy(j)]*theta;
end
end
mesh(xx,yy,zz);
title('Result of Solving Normal Equation');
legend('Training data', 'Linear regression');         


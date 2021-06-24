


data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2); %all lines, column 1 for x,2 for y
m = length(y); % number of training examples

%%%% Plot Data

plot(X,y, 'rx', 'MarkerSize', 10);
xlabel('Population in 10*10^3'); 
ylabel('Profit in in $10*10^3'); 

%%%% Gradient descent

%gradient descent settings
iterations = 1500;
alpha = 0.01;

%adjustments
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters


%cost function without gradient descent
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]  Cost computed = %f\n', J);

%running gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);
fprintf('Theta found by gradient descent:\n');
fprintf('theta0 = %.2f and theta1= %.2f\n\n', theta(1),theta(2));
J = computeCost(X, y, theta);
fprintf('With theta0 = %.2f and theta1= %.2f  Cost computed = %f\n', theta(1),theta(2),J);

%plot the linear fit
hold on; 
h = X*theta;
plot(X(:,2), h, '-');
legend('Training data', 'Linear regression');
hold off ;


%%%predict values
pop1 = 40000;
pop2 = 100000;
predict1 = [1, pop1] *theta;
fprintf('For population = 35,000, we predict a profit of $%.2f\n',...
    predict1);
predict2 = [1, pop2] * theta;
fprintf('For population = 70,000, we predict a profit of $%.2f\n',...
    predict2);



%extracted from the ex1;
%%% Visualizing J(theta_0, theta_1)
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end



J_vals = J_vals';

figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');


figure;

contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


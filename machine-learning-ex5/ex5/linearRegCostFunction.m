function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


%--------- CORRECT
%h = sigmoid(X*theta);
%Reg = 0;
%j = 2;
%while j <= length(theta)
%    Reg = Reg + (lambda/(2*m))* theta(j)^2;
%j= j+1;
%end
%
%i=1;
%while i<= m
%    J = J+ 1/(2*m)* (X(i,:)*theta- y(i))^2;
%    i=i+1;
%    end;
%J = J+Reg;
%------------------------
h = X*theta;

% regularize theta by removing first value
theta_reg = [0;theta(2:end, :);];
J = (1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*theta_reg'*theta_reg;

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);














% =========================================================================

grad = grad(:);

end

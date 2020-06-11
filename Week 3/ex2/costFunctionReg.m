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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h0 = sigmoid(X * theta);
error = -y .* log(h0) - (1-y) .* log(1-h0);
theta_cost = theta(2:end,:,1)
theta_grad = theta
theta_grad(1) = 0
regular = lambda/(2*m)* sum(theta_cost.^2)
grad = 1/m * X' * (h0-y) + (lambda/m)*theta_grad;
J = 1/m * sum(error) + regular;





% =============================================================

end

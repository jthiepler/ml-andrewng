function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
results = zeros(64,3)

for i = 1:8
  
  if rem(i,2) == 0
    c = 0.003 * 10^(i/2) 
  endif
  if rem(i,2) ~= 0
    c = 0.01 * 10^((i/2)-0.5) 
  endif
  
  for j = 1:8
    
    if rem(j,2) == 0
      s = 0.003 * 10^(j/2) 
    endif
    if rem(j,2) == 1
      s = 0.01 * 10^((j/2)-0.5) 
    endif
    
    p = svmPredict(svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s)), Xval);
    results((i-1)*8 + j,1) = c
    results((i-1)*8 + j,2) = s
    results((i-1)*8 + j,3) = mean(double(p ~= yval))
    [v, k] = min(results(:,3))
    C = results(k, 1)
    sigma = results(k, 2)
    
  endfor
endfor




% =========================================================================

end

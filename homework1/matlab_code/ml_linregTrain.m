function [w, beta]  = ml_linregTrain(t, X, lambda) 
%function [w, beta]  = ml_linregTrain(t, X, lambda) 
%  
% What it does: It trains a linear regression model with regularization
%
% Inputs: 
%         t: N x 1 output data vector
%         X: N x (D+1) input data vectro with ones alreadt added in the first column
%         lambda: the positive regularizarion parameter 
% Outputs: 
%         w: the trained D+1 dimensional of the parameters    
%         beta: the trained inverse variance parameter 
%
% Michalis Titsias (2014)


% X assumed already to have its first colunm filled with ones 
[N, D] = size(X);

% Do the job
T = X'*t;
K = X'*X  + lambda*eye(D);
w = K\T;

% Model response/predictions (needed for computing beta)
y = X*w;  

% Beta
beta = N/sum((y-t).^2);





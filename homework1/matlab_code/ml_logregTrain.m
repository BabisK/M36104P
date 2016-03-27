function w = ml_logregTrain(t, X, lambda, winit, options)
%function w = ml_logregTrain(t, X, lambda, winit, options)
%
% What it does: It trains using gradient ascend a linear logistic regression  
%               model with regularization
%
% Inputs: 
%         t: N x 1 binary output data vector indicating the two classes
%         X: N x (D+1) input data vector with ones already added in the first column
%         lambda: the positive regularizarion parameter
%         winit: D+1 dimensional vector of the initial values of the parameters 
%         options: options(1) is the maximum number of iterations 
%                  options(2) is the tolerance
%                  options(3) is the learning rate eta 
% Outputs: 
%         w: the trained D+1 dimensional vector of the parameters    
%  
% Michalis Titsias (2014)

w = winit;

% Maximum number of iteration of gradient ascend
iter = options(1); 

% Tolerance
tol = options(2);

% Learning rate
eta = options(3);
 
Ewold = -Inf; 
for it=1:iter
%    
    yx = X*w;
    s = sigmoid(yx);
    
    % Compute the cost function to check convergence
    % sxolio: this is not computed in a numerical stable way for paidagogical reasons
    Ew = sum( t.*log(s)  + (1-t).*log(1-s) )  - (0.5*lambda)*(w'*w);
    
    % Show the current cost function on screen
    fprintf('Iteration: %d, Cost function: %f\n',it, Ew); 

    % Break if you achieve the desired accuracy in the cost function
    if abs(Ew - Ewold) < tol 
        break;
    end

    % Gradient 
    gradient = X'*(t - s) - lambda*w;
    
    % Update parameters based on gradient ascend 
    w = w + eta*gradient; 
    
    Ewold = Ew; 
%
end
        
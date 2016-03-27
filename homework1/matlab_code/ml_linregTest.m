function [ytest, vtest]  = ml_linregTest(w, beta, Xtest) 
%function [ytest, vtest]  = ml_linregTest(w, beta, Xtest) 
%  
% What it does: It tests an already trained linear regression model
%
% Inputs: 
%         w: the D+1 dimensional vector of the parameters  
%         beta:  the inverse variance parameter 
%         Xtest: Ntest x (D+1) input test data with ones already added in the first column 
% Outputs: 
%         ytest: Ntest x 1 vector of mean predictions      
%         vtest: Ntest x 1 vector of predictive variances   
%
% Michalis Titsias (2014)


Ntest = size(Xtest,1); 

% Mean predictions
ytest = Xtest*w;

% Variances 
vtest = (1/beta)*ones(Ntest,1); 



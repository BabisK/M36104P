function [ttest, ytest]  = ml_logregTest(w, Xtest) 
%function [ttest, ytest]  = ml_logregTest(w, Xtest) 
%  
% What it does: It tests an already trained logistic regression model
%
% Inputs: 
%         w: the D+1 dimensional vector of the parameters   
%         Xtest: Ntest x (D+1) input test data with ones already added in the first column 
% Outputs: 
%         test: the predicted class labels
%         ytest: Ntest x 1 vector of the sigmoid probabilities     
%
% Michalis Titsias (2014)

% Mean predictions
ytest = sigmoid(Xtest*w);

% Hard classification decisions 
ttest = round(ytest); 


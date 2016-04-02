% DEMO: LOGISTIC REGRESSION 

clear all;
close all; 

% Load the training and test data   
train_data = load('data2Tr.txt');
X = train_data(:,1:2); % inputs 
t = train_data(:,3); % outputs 
Xtest = load('data2Ts.txt');


% Number and dimension of training data 
[N D] = size(X);

% Number of test data (the dimension must be obviously the same)
Ntest = size(Xtest,1);

% Add 1 as the first for both the training input and test inputs 
X = [ones(N,1), X];
Xtest = [ones(Ntest,1), Xtest]; 


% Initila w for the gradient ascent
winit = zeros(D+1,1);

% Regularization parameter lambda 
lambda = 0; 

% Maximum number of iterations of the gradient ascend
options(1) = 500; 
% Tolerance 
options(2) = 1e-6; 
% Learning rate 
options(3) = 8/N;  

% Train the model 
w = ml_logregTrain(t, X, lambda, winit, options); 

% Test the model 
[ttest, ytest]  = ml_logregTest(w, Xtest); 


% Do two plots:  the first plots the training data and the decision boundaries 
% with the 0.1 and 0.9 parallel lines  

figure; 
plot(X(t==1,2),X(t==1,3),'.','Markersize',20);
hold on; 
plot(X(t==0,2),X(t==0,3),'r.','Markersize',20);

minX = min(X(:,2)); 
maxX = max(X(:,2));
t1 = -w(1)/w(3)  - (w(2)/w(3))*[minX maxX]
t2 = -log(0.9/0.1)/w(3) - w(1)/w(3)  - (w(2)/w(3))*[minX maxX]
t3 = -log(0.1/0.9)/w(3) - w(1)/w(3)  - (w(2)/w(3))*[minX maxX]
plot([minX maxX], -w(1)/w(3)  - (w(2)/w(3))*[minX maxX],'k','LineWidth',2);
plot([minX maxX], -log(0.9/0.1)/w(3) - w(1)/w(3)  - (w(2)/w(3))*[minX maxX],'k-.','LineWidth',1);
plot([minX maxX], -log(0.1/0.9)/w(3) - w(1)/w(3)  - (w(2)/w(3))*[minX maxX],'k-.','LineWidth',1);


% In the second we plot the probabilities in the test data 
figure;
hold on;
for n=1:Ntest
  plot(Xtest(n,2),Xtest(n,3),'.','Markersize',20, 'Color', [1-ytest(n) 0 ytest(n)]);
end

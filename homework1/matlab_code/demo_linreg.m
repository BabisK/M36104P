% DEMO: LINEAR REGRESSION

clear all;
close all; 

% load the training and test data   
train_data = load('data1Tr.txt');
X = train_data(:,1); % inputs 
t = train_data(:,2); % outputs 
Xtest = load('data1Ts.txt');


% plot the data to see them!
plot(X,t,'+b','Markersize',10); 
hold on;


% Number and dimension of training data 
[N D] = size(X);

% Number of test data (the dimension must be obviously the same)
Ntest = size(Xtest,1);

% Add 1 as the first for both the training input and test inputs 
X = [ones(N,1), X];
Xtest = [ones(Ntest,1), Xtest]; 

% Regularizarion parameter
lambda = 0; 

% Train the model 
[w, beta]  = ml_linregTrain(t, X, lambda);

% Test the model 
[ytest, vtest]  = ml_linregTest(w, beta, Xtest); 

% Plot the mean test predictions and the 1-std uncertainties
plot(Xtest(:,2),ytest,'r','Linewidth',2);
plot(Xtest(:,2),ytest-sqrt(vtest),'r--','Linewidth',1);
plot(Xtest(:,2),ytest+sqrt(vtest),'r--','Linewidth',1);





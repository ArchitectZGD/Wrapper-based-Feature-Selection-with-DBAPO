% --------- General configuration ---------
N = 20;         % Population size (20)
max_FEs = 2000; % Maximum fitness evaluation number (2000)
max_Run = 20;    % Maximum simulation run (20 Runs)
ho = 0.2;       % Outer held-out test set ratio (20% of data reserved for final test evaluation)
ho_inner = 0.2; % Inner validation set ratio (20% split inside training set for fitness scoring)
CR = 0.9;       % Crossover rate (0.9), function-specific parameter value
FITNESS_ALPHA = 0.99;   % Fitness weight, cost = alpha*error + (1-alpha)*ratio; set to 0.95 for robustness experiments

% --------- Classifier selection ---------
% Options: 'KNN' | 'SVM' | 'MLP' | 'LR' | 'RF', applied globally and shared by all algorithms
global classifier
classifier = 'KNN' ;

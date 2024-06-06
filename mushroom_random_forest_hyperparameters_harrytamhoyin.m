% Import the mushroom poisonous file
mushroom_raw = readtable('agaricus-lepiota.csv', 'ReadVariableNames', false);

% Replace the header
header = {'poisonous' 'cap_shape' 'cap_surface' 'cap_color' 'bruises' 'odor'...
    'gill_attachment' 'gill_spacing' 'gill_size' 'gill_color' 'stalk_shape'... 
    'stalk_root' 'stalk_surface_above_ring' 'stalk_surface_below_ring'... 
    'stalk_color_above_ring' 'stalk_color_below_ring' 'veil_type' 'veil_color'... 
    'ring_number' 'ring_type' 'spore_print_color' 'population' 'habitat'};
mushroom_raw.Properties.VariableNames = header;

% Remove the column stalk root ('e.1') and veil type (p.1)
mushroom_table = removevars(mushroom_raw, {'stalk_root', 'veil_type'});

% Change the 'table' data type to 'double' data type of the dataset
mushroom_cat = categorical(mushroom_table{:,:});
mushroom_numeric = double(mushroom_cat);

% Split the data into features and label
% Features
X = mushroom_numeric(:, 2:end);
% Label
Y = mushroom_numeric(:, 1);

% Set the random seed
rng(42);

% Split the data into 80% of training set and 20% of test set
cv = cvpartition(Y, 'Holdout', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_test = X(test(cv), :);
Y_test = Y(test(cv), :);

% Train the random forest with the training set and find the
% hyperparameters
trees = templateTree('Reproducible', true);
rf = fitcensemble(X_train, Y_train, 'Method', 'bag', 'Learners', trees, ...
    'OptimizeHyperparameters', {'NumLearningCycle','NumVariablesToSample'}, ...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch'));

% Save the training file
save('mushroom_random_forest_hyperparameters_harrytamhoyin.mat', 'rf')
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

% Create 10 fold cross validation on the training set
cv_train = cvpartition(Y_train, 'KFold', 10);

% Array to store the confusion matrix, accuracy, precision, recall, 
% specificity, F1 score, AUC value and training time of validation set for each fold
cm_matrix = zeros(2,2);
cv_accuracy = zeros(10, 1);
cv_precision = zeros(10, 1);
cv_recall = zeros(10, 1);
cv_specificity = zeros(10, 1);
cv_f1score = zeros(10, 1);
cv_AUC = zeros(10,1);
cv_training_time = zeros(10,1);

% Array to store best model and best AUC value
best_nb_multinomial = [];
best_nb_auc = 0;

% Create a figure
figure;
hold on;

% Perform 10 fold cross valiadation
for fold = 1:10

    % Training indices and validation indices for the current fold
    train_indices = training(cv_train, fold);
    validation_indices = test(cv_train, fold);
    
    % Split the training data into training data and validation data for the
    % current fold
    X_train_fold = X_train(train_indices, :);
    Y_train_fold = Y_train(train_indices, :);
    X_validation_fold = X_train(validation_indices, :);
    Y_validation_fold = Y_train(validation_indices, :);

    % Record the start time for training
    start_time_fold = tic;

    % Train the multivariate multinomial Naive Bayes model on the current training fold
    nb_multinomial = fitcnb(X_train_fold, Y_train_fold, 'ClassNames', unique(Y_train_fold),...
        'CategoricalPredictors', (1:20), 'DistributionNames', 'mvmn');
    
    % Record the training time for the current fold
    cv_training_time(fold) = toc(start_time_fold);

    % Use the validation set for prediction
    [predictions_fold, prob_estimates_val] = predict(nb_multinomial, X_validation_fold);
    
    % Confusion matrix for the current fold
    cm_fold = confusionmat(Y_validation_fold, predictions_fold); 

    % Accumulate the confusion matrix
    cm_matrix = cm_matrix + cm_fold;

    % Calculate and store the performance metrics for the current fold
    cv_accuracy(fold) = sum(diag(cm_fold))/sum(cm_fold(:));
    cv_precision(fold) = cm_fold(2,2)/sum(cm_fold(:,2));
    cv_recall(fold) = cm_fold(2,2)/sum(cm_fold(2,:));
    cv_specificity(fold) = cm_fold(1,1)/sum(cm_fold(:,1));
    cv_f1score(fold) = 2*(cv_precision(fold)*cv_recall(fold))/(cv_precision(fold)+cv_recall(fold)); 
    
    % Calculate and store AUC value for the current fold
    % Note that poisonous is 14 and edible is 5
    [X_val_fold, Y_val_fold, T_val_fold, AUC_val_fold] = perfcurve(Y_validation_fold, prob_estimates_val(:,2), '14');
    cv_AUC(fold) = AUC_val_fold; 
    
    % Plot the ROC curve
    plot(X_val_fold, Y_val_fold, 'DisplayName', ['Fold ' num2str(fold) ' (AUC = ' num2str(AUC_val_fold) ')']);

    % Check if the current fold model is the best model with the highest
    % AUC value
    if AUC_val_fold > best_nb_auc
        best_nb_auc = AUC_val_fold;
        best_nb_multinomial = nb_multinomial;
    
    end
end

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve For Naive Bayes (10 Fold Cross Validation)');
legend('show');

% Calculate the average performance metrics, average AUC score, total training time 
% and average training time for the ten fold
avg_cv_accuracy = mean(cv_accuracy);
avg_cv_precision = mean(cv_precision);
avg_cv_recall = mean(cv_recall);
avg_cv_specificity = mean(cv_specificity);
avg_cv_f1score = mean(cv_f1score);
avg_cv_AUC = mean(cv_AUC);
total_cv_training_time = sum(cv_training_time);
avg_cv_training_time = mean(cv_training_time);

% Display the average metrics, total training time and average training time for validation set
disp(['Average Accuracy (Validation Set): ' num2str(avg_cv_accuracy)]);
disp(['Average Precision (Validation Set): ' num2str(avg_cv_precision)]);
disp(['Average Recall (Validation Set): ' num2str(avg_cv_recall)]);
disp(['Average Specificity (Validation Set): ' num2str(avg_cv_specificity)]);
disp(['Average F1 Score (Validation Set): ' num2str(avg_cv_f1score)]);
disp(['Average AUC (Validation Set): ' num2str(avg_cv_AUC)]);
disp(['Total Training Time: ' num2str(total_cv_training_time) ' seconds']);
disp(['Average Training Time: ' num2str(avg_cv_training_time) ' seconds']);

% Display the confusion matrix chart
% Note that 1 is edible and 2 is poisonous for this confusion matrix chart
figure;
confusionchart(cm_matrix, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix For Naive Bayes (10 Fold Cross Validation)')

% Display the best model with the highest AUC value
disp(['Best AUC value: ' num2str(best_nb_auc)]);

% Save the best model
save('mushroom_naive_bayes_training_harrytamhoyin.mat', 'best_nb_multinomial')
% ATTENTION: PLEASE RUN THE TRAINING SCRIPT (MUSHROOM_NAIVE_BAYES_TRAINING_HARRYTAMHOYIN) FIRST
% Load the training script for the test data and the best trained
% multivariate multinomial Naive Bayes model
load('mushroom_naive_bayes_training_harrytamhoyin.mat')

% Apply the trained multivariate multinomial Naive Bayes model on test set for prediction and
% count the test time
% Note if there is error for X_test, please run the training script first
test_start_time = tic;
[predictions_test, prob_estimates] = predict(best_nb_multinomial, X_test);
test_time = toc(test_start_time);

% Calculate AUC value for the test set
[X_val_test, Y_val_test, T_val_test, AUC_val_test] = perfcurve(Y_test, prob_estimates(:, 2), '14');

% Plot the ROC curve
figure;
plot(X_val_test, Y_val_test, 'DisplayName', ['Test Set: ' '(AUC = ' num2str(AUC_val_test) ')']);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve For Naive Bayes (Test Set)');
legend('show');

% Obtain confusion matrix and calculate the performance metrics
cm_test = confusionmat(Y_test, predictions_test);
test_accuracy = sum(diag(cm_test))/sum(cm_test(:));
test_precision = cm_test(2,2)/sum(cm_test(:,2));
test_recall = cm_test(2,2)/sum(cm_test(2,:));
test_specificity = cm_test(1,1)/sum(cm_test(:,1));
test_f1score = 2*(test_precision*test_recall)/(test_precision+test_recall); 

% Display the performance metrics, AUC value and test time for test set
disp(['Test Accuracy (Test Set): ' num2str(test_accuracy)]);
disp(['Test Precision (Test Set): ' num2str(test_precision)]);
disp(['Test Recall (Test Set): ' num2str(test_recall)]);
disp(['Test Specificity (Test Set): ' num2str(test_specificity)]);
disp(['Test F1 Score (Test Set): ' num2str(test_f1score)]);
disp(['Test AUC value (Test Set): ' num2str(AUC_val_test)]);
disp(['Test Time: ' num2str(test_time) ' seconds']);

% Display the confusion matrix chart
figure;
confusionchart(Y_test, predictions_test, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix For Naive Bayes (Test Set)');
tic
% 关闭警告信息，关闭所有图窗，清空变量，清空命令行
warning off;
close all;
clear;
clc;

% 使用最优参数进行最终评估
threshold_factor =0.0043231;
std_weight =0.60276;
alpha = 0.38378;
k =18;
R =0.41784;

% 加载数据并检查
load('train_interval_data_with_labels.mat');
normData_train = train_normData;
stdVals_train = repmat(train_stdVals, size(normData_train, 1), 1);

load('test_interval_data_with_labels.mat');
normData_test = test_normData;
stdVals_test = repmat(test_stdVals, size(normData_test, 1), 1);
test_labels = test_labels;

% 最终评估
[totalAccuracy, positiveAccuracy, negativeAccuracy, F1Score_positive, F1Score_negative, balancedF1Score] = evaluateModel(threshold_factor, std_weight, alpha, k, R, normData_train, stdVals_train, normData_test, stdVals_test, test_labels);

fprintf('正类数据分类F1-score：%.2f%%\n', F1Score_positive * 100);
fprintf('负类数据分类F1-score：%.2f%%\n', F1Score_negative * 100);
fprintf('基于F1-score的balanced F1-score：%.2f%%\n', balancedF1Score * 100);
 toc
% 计算分类准确率和F1-score
function [totalAccuracy, positiveAccuracy, negativeAccuracy, F1Score_positive, F1Score_negative, balancedF1Score] = evaluateModel(threshold_factor, std_weight, alpha, k, R, train, train_stdVals, test, test_stdVals, test_labels)
    % 计算训练数据的距离矩阵
    N_train = size(train, 1);
    distances_train = zeros(N_train, N_train);
    for i = 1:N_train
        for j = i+1:N_train
            distances_train(i,j) = uncertainDistance(train(i,:), train(j,:), train_stdVals(i,:), train_stdVals(j,:), std_weight);
            distances_train(j,i) = distances_train(i,j); % 距离是对称的
        end
    end

    % 从训练集距离矩阵确定阈值
    threshold = median(distances_train(:)) * threshold_factor;

    % 计算测试集数据点与训练集数据点间的最小距离
    N_test = size(test, 1);
    min_distances_test = zeros(N_test, 1);
    for i = 1:N_test
        distances = arrayfun(@(j) uncertainDistance(test(i,:), train(j,:), test_stdVals(i,:), train_stdVals(j,:), std_weight), 1:N_train);
        min_distances_test(i) = min(distances);
    end

    % 检测测试集异常
    isOutlier_test = min_distances_test > threshold;

    % 计算测试集分类正确率
    normal_indices = test_labels == -1;
    outlier_indices = test_labels ~= -1;

    true_negative = sum(isOutlier_test(normal_indices) == 0);
    false_positive = sum(isOutlier_test(normal_indices) == 1);
    false_negative = sum(isOutlier_test(outlier_indices) == 0);
    true_positive = sum(isOutlier_test(outlier_indices) == 1);

    % 计算精确率、召回率和F1-score
    precision_normal = true_negative / (true_negative + false_negative);
    recall_normal = true_negative / (true_negative + false_positive);
    F1_score_normal = 2 * (precision_normal * recall_normal) / (precision_normal + recall_normal);

    precision_outlier = true_positive / (true_positive + false_positive);
    recall_outlier = true_positive / (true_positive + false_negative);
    F1_score_outlier = 2 * (precision_outlier * recall_outlier) / (precision_outlier + recall_outlier);

    % 计算基于F1-score的balanced F1-score
    balanced_F1_score = (F1_score_normal + F1_score_outlier) / 2;

    totalAccuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative);
    positiveAccuracy = precision_outlier;
    negativeAccuracy = precision_normal;
    F1Score_positive = F1_score_outlier;
    F1Score_negative = F1_score_normal;
    balancedF1Score = balanced_F1_score;
end

% 不确定距离计算函数定义
function d = uncertainDistance(xi, xj, sigma_i, sigma_j, std_weight)    
    % 计算基于标准差加权的欧式距离
    d = sqrt(sum(((xi - xj).^2) ./ (std_weight * sigma_i.^2 + (1 - std_weight) * sigma_j.^2)));
end

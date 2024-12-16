tic
% 关闭警告信息，关闭所有图窗，清空变量，清空命令行
warning off;
close all;
clear;
clc;

% 加载数据
load('train_interval_data_with_labels.mat');
train = train_normData; % 均值数据
train_labels = train_labels;
train_stdVals = train_stdVals;  % 每个特征的标准差

load('test_interval_data_with_labels.mat');
test = test_normData; % 均值数据
test_labels = test_labels;
test_stdVals = test_stdVals;  % 每个特征的标准差

% 构造协方差矩阵
train_Sigma = diag(train_stdVals.^2);  % 单一协方差矩阵，适用于所有训练数据点
test_Sigma = diag(test_stdVals.^2);    % 单一协方差矩阵，适用于所有测试数据点

% 已知的最优参数
bestParams.k = 171; % 例如，最优参数k的值
bestParams.gamma =   0.48062 ; % 例如，最优参数gamma的值
bestParams.scaleFactor = 0.10005 ; % 例如，最优参数scaleFactor的值
bestParams.g = 0.92031 ; % 例如，最优参数g的值
bestParams.f =1.9789; % 例如，最优参数f的值

% 使用最优参数在测试集上进行评估并输出结果
[meanULOFScore, F1Score_positive, F1Score_negative, balancedF1Score] = evaluateOnTest(bestParams.k, bestParams.gamma, bestParams.scaleFactor, bestParams.g, bestParams.f, train, test, test_labels, train_Sigma, test_Sigma);
fprintf('最优参数：\n');
disp(bestParams);
fprintf('平均 ULOF 分数: %.4f\n', meanULOFScore);
fprintf('正类 F1-score: %.2f%%\n', F1Score_positive * 100);
fprintf('负类 F1-score: %.2f%%\n', F1Score_negative * 100);
fprintf('基于 F1-score 的 balanced F1-score: %.2f%%\n', balancedF1Score * 100);
toc
% 在测试集上验证模型性能
function [meanULOFScore, F1Score_positive, F1Score_negative, balancedF1Score] = evaluateOnTest(k, gamma, scaleFactor, g, f, train, test, test_labels, train_Sigma, test_Sigma)
    num_samples_train = size(train, 1);
    num_samples_test = size(test, 1);
    ULOF_scores_test = zeros(num_samples_test, 1);
    k = round(k); % 确保k是整数

    for i = 1:num_samples_test
        distances = zeros(num_samples_train, 1);
        for j = 1:num_samples_train
            distances(j) = uncertain_distance(test(i, :), train(j, :), test_Sigma, f);
        end
        [~, idx] = sort(distances);
        k_nearest = distances(idx(1:k));
        local_density = mean(1 ./ (k_nearest .^ g));
        ULOF_scores_test(i) = local_density;
    end

    threshold = scaleFactor;
    predicted_labels = ULOF_scores_test > threshold;
    actual_labels = test_labels > 0;

    % 计算正类和负类的F1-score
    TP = sum((predicted_labels == 1) & (actual_labels == 1));
    FP = sum((predicted_labels == 1) & (actual_labels == 0));
    TN = sum((predicted_labels == 0) & (actual_labels == 0));
    FN = sum((predicted_labels == 0) & (actual_labels == 1));

    Precision_positive = TP / (TP + FP);
    Recall_positive = TP / (TP + FN);
    F1Score_positive = 2 * (Precision_positive * Recall_positive) / (Precision_positive + Recall_positive);

    Precision_negative = TN / (TN + FN);
    Recall_negative = TN / (TN + FP);
    F1Score_negative = 2 * (Precision_negative * Recall_negative) / (Precision_negative + Recall_negative);

    % 计算基于F1-score的balanced F1-score
    balancedF1Score = (F1Score_positive + F1Score_negative) / 2;

    meanULOFScore = mean(ULOF_scores_test);
end

% 不确定距离计算函数定义
function d = uncertain_distance(x, y, Sigma, f)
    diff = x - y; % 特征差
    if size(diff, 1) == 1
        diff = diff'; % 确保diff是列向量
    end
    d = sqrt(diff' * (Sigma \ diff)) * f; % 马氏距离乘以不确定性因子
end

% ULOF计算函数
function ulrd = ulrd(x, k, Sigma, g, neighbors)
    k_dist = neighbors(k).dist;
    sum_reachability = 0;
    for i = 1:length(neighbors)
        reach_dist = max(k_dist, uncertain_distance(x, neighbors(i).point, Sigma, 1));
        sum_reachability = sum_reachability + reach_dist;
    end
    ulrd = length(neighbors) / sum_reachability;
end

function ulof_val = ulof(x, neighbors, k, Sigma, g)
    ulrd_o = ulrd(x, k, Sigma, g, neighbors);
    sum_ulrd_ratios = 0;
    for i = 1:length(neighbors)
        neighbor_ulrd = ulrd(neighbors(i).point, k, Sigma, g, neighbors(i).neighbors);
        sum_ulrd_ratios = sum_ulrd_ratios + (neighbor_ulrd / ulrd_o);
    end
    ulof_val = sum_ulrd_ratios / length(neighbors);
end

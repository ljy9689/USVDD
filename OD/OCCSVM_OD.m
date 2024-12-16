tic
% 关闭警告信息，关闭所有图窗，清空变量，清空命令行
warning off;
close all;
clear all;
clc;

% 已知的最优参数
kernelScale =1.7888; % 例如，最优参数kernelScale的值
num_points_per_interval = 6; % 例如，最优参数num_points_per_interval的值

% 加载数据
load('train_interval_data_with_labels.mat');
train = train_intervalData;
train_labels = train_labels;

load('test_interval_data_with_labels.mat');
test = test_intervalData;
test_labels = test_labels;

m = 10; % 区间对应的特征数

% 提取区间数据的维度信息
[num_samples, num_features_total] = size(train);
num_features = num_features_total / 2;

% 将区间值数据转换为具有不确定权重的精确数据点
train_data = [];
train_expanded_labels = [];
train_expanded_weights = [];

for i = 1:num_samples
    lower_bound = train(i, 1:num_features);
    upper_bound = train(i, num_features + 1:num_features * 2);
    [points, weights] = generate_points_within_interval(lower_bound, upper_bound, num_points_per_interval);
    train_data = [train_data; points];
    train_expanded_labels = [train_expanded_labels; repmat(train_labels(i), num_points_per_interval, 1)];
    train_expanded_weights = [train_expanded_weights; weights];
end

% 对测试集进行相同处理
[num_samples_test, ~] = size(test);
test_data = [];
test_expanded_labels = [];

for i = 1:num_samples_test
    lower_bound = test(i, 1:num_features);
    upper_bound = test(i, num_features + 1:num_features * 2);
    [points, ~] = generate_points_within_interval(lower_bound, upper_bound, num_points_per_interval);
    test_data = [test_data; points];
    test_expanded_labels = [test_expanded_labels; repmat(test_labels(i), num_points_per_interval, 1)];
end

% 定义核函数和SVM参数
kernelFunction = 'rbf'; % 核函数类型
boxConstraint = 1; % 固定盒约束参数为1

% 训练带权重的 SVM
SVMModel = fitcsvm(train_data, train_expanded_labels, ...
    'KernelFunction', kernelFunction, 'BoxConstraint', boxConstraint, ...
    'KernelScale', kernelScale, 'Weights', train_expanded_weights, ...
    'Standardize', true, 'ClassNames', [-1, 1]);

% 在测试集上进行预测
[predicted_labels, ~] = predict(SVMModel, test_data);

% 计算分类正确率
totalAccuracy = sum(predicted_labels == test_expanded_labels) / numel(test_expanded_labels);
positiveAccuracy = sum(predicted_labels(test_expanded_labels == 1) == test_expanded_labels(test_expanded_labels == 1)) / numel(test_expanded_labels(test_expanded_labels == 1));
negativeAccuracy = sum(predicted_labels(test_expanded_labels == -1) == test_expanded_labels(test_expanded_labels == -1)) / numel(test_expanded_labels(test_expanded_labels == -1));

% 计算测试集预测后的统计指标
TP = sum((predicted_labels == 1) & (test_expanded_labels == 1));
FP = sum((predicted_labels == 1) & (test_expanded_labels == -1));
TN = sum((predicted_labels == -1) & (test_expanded_labels == -1));
FN = sum((predicted_labels == -1) & (test_expanded_labels == 1));

% 计算 Precision 和 Recall
Precision_positive = TP / (TP + FP);
Recall_positive = TP / (TP + FN);
F1Score_positive = 2 * (Precision_positive * Recall_positive) / (Precision_positive + Recall_positive);

Precision_negative = TN / (TN + FN);
Recall_negative = TN / (TN + FP);
F1Score_negative = 2 * (Precision_negative * Recall_negative) / (Precision_negative + Recall_negative);

% 计算基于 F1-score 的 balanced F1-score
balancedF1Score = (F1Score_positive + F1Score_negative) / 2;

% 输出结果
fprintf('已知最优参数：\n');
fprintf('Kernel Scale: %.4f\n', kernelScale);
fprintf('Num Points per Interval: %d\n', num_points_per_interval);
fprintf('总体分类正确率: %.2f%%\n', totalAccuracy * 100);
fprintf('正类 F1-score: %.2f%%\n', F1Score_positive * 100);
fprintf('负类 F1-score: %.2f%%\n', F1Score_negative * 100);
fprintf('基于 F1-score 的 balanced F1-score: %.2f%%\n', balancedF1Score * 100);
toc
% 函数定义：生成区间内的点并根据位置调整权重
function [points, weights] = generate_points_within_interval(lower, upper, num_points)
    dim = length(lower);
    points = zeros(num_points, dim);
    weights = zeros(num_points, 1);

    % 在区间内生成点并根据生成的位置调整权重
    for i = 1:num_points
        points(i, :) = lower + (upper - lower) .* rand(1, dim);
        % 假设中点附近的点权重更高
        center = (lower + upper) / 2;
        distance = norm(points(i, :) - center);
        weights(i) = 1 / (1 + distance); % 使用距离的倒数作为权重
    end
    % 归一化权重
    weights = weights / sum(weights);
end

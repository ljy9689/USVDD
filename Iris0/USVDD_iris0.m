tic
% 关闭警告信息，关闭所有图窗，清空变量，清空命令行
warning off;
close all;
clear;
clc;

% 加载数据
load('train_interval_data_with_labels.mat');
train = train_intervalData;
train_labels = train_labels;
train_stdVals = train_stdVals;


% 使用最优参数
C =   8.7148;
sigma =  7.7267;
aa = [ 0.71223    0.87564    0.99066    0.99006];

% 加载测试集数据
load('test_interval_data_with_labels.mat');
test = test_intervalData;
test_labels = test_labels;
test_stdVals = test_stdVals;

m = 4; % 特征个数

% 计算逆函数
Xin = zeros(size(train, 1), m);
X1in = zeros(size(test, 1), m);
for i = 1:m
    Xin(:, i) = train(:, i) + ((train_stdVals(i) * sqrt(3)) / pi) * log(aa(i) / (1 - aa(i)));
    X1in(:, i) = test(:, i) + ((test_stdVals(i) * sqrt(3)) / pi) * log(aa(i) / (1 - aa(i)));
end

% 核函数参数
sigma_sq_inv = 1 / (2 * sigma^2);

% 计算核函数矩阵
K = exp(-pdist2(Xin, Xin).^2 .* sigma_sq_inv);

% 设置二次规划的参数
H = K;
f = -diag(K);
Aeq = ones(1, size(train, 1));
beq = 1;
lb = zeros(size(train, 1), 1);
ub = C * ones(size(train, 1), 1);
options = optimoptions('quadprog', 'Display', 'off');
a = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);

if isempty(a)
    error('Optimization failed to find a solution.');
end

a(a < 1e-5) = 0;

% 计算测试集预测
K_test = exp(-pdist2(X1in, Xin).^2 .* sigma_sq_inv);
predict = sign(K_test * a - mean(K_test * a));

% 计算准确率
overall_accuracy = mean(predict == test_labels);
positive_accuracy = mean(predict(test_labels == 1) == test_labels(test_labels == 1));
negative_accuracy = mean(predict(test_labels == -1) == test_labels(test_labels == -1));

% 计算测试集预测后的统计指标
TP = sum((predict == 1) & (test_labels == 1));
FP = sum((predict == 1) & (test_labels == -1));
TN = sum((predict == -1) & (test_labels == -1));
FN = sum((predict == -1) & (test_labels == 1));

% 计算 Precision 和 Recall
Precision = TP / (TP + FP);
Recall = TP / (TP + FN);

% 计算正类和负类的 F1-score
F1Score_positive = 2 * (Precision * Recall) / (Precision + Recall);

Precision_negative = TN / (TN + FN);
Recall_negative = TN / (TN + FP);
F1Score_negative = 2 * (Precision_negative * Recall_negative) / (Precision_negative + Recall_negative);

% 计算基于 F1-score 的 balanced accuracy
balancedF1Score = (F1Score_positive + F1Score_negative) / 2;

% 输出结果

fprintf('整体分类正确率: %.2f%%\n', overall_accuracy * 100);
fprintf('正类（标签 1）分类正确率: %.2f%%\n', positive_accuracy * 100);
fprintf('负类（标签 -1）分类正确率: %.2f%%\n', negative_accuracy * 100);
fprintf('正类 F1 分数: %.2f%%\n', F1Score_positive * 100);
fprintf('负类 F1 分数: %.2f%%\n', F1Score_negative * 100);
fprintf('基于 F1-score 的 balanced accuracy: %.2f%%\n', balancedF1Score * 100);
toc
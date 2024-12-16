tic
% 关闭警告、关闭图窗、清空变量、清空命令行
warning off;
close all;
clear;
clc;

% 加载训练和测试数据
load('train_interval_data_with_labels.mat');
train = train_intervalData;
train_labels = train_labels;

load('test_interval_data_with_labels.mat');
test = test_intervalData;
test_labels = test_labels;

C = 10.16 ; % 软间隔数值
sigma =0.0061454; % 核函数参数
nam = 0.46005; % 半宽因子

m = 10; % 特征个数

% 计算训练集和测试集的区间中值和半宽度
Xmid = (train(:, 1:2:end) + train(:, 2:2:end)) / 2;
Xwid = (train(:, 2:2:end) - train(:, 1:2:end)) / 2;
Xmid1 = (test(:, 1:2:end) + test(:, 2:2:end)) / 2;
Xwid1 = (test(:, 2:2:end) - test(:, 1:2:end)) / 2;

% 初始化矩阵以存储核函数值
K = zeros(size(train, 1), size(train, 1));
K1 = zeros(size(test, 1), size(train, 1));

% 计算训练集的核函数
for i = 1:size(train, 1)
    for j = 1:size(train, 1)
        dist = sum(nam * (Xmid(i, :) - Xmid(j, :)).^2 + ...
            (1 - nam) * (Xwid(i, :) - Xwid(j, :)).^2);
        K(i, j) = exp(-dist / (2 * sigma^2));
    end
end

% 目标函数和约束条件的设置
H = 2 * K; % 对称矩阵
f = -ones(size(train, 1), 1);
A = [];
b = [];
Aeq = ones(1, size(train, 1));
beq = 1;
lb = zeros(size(train, 1), 1);
ub = C * ones(size(train, 1), 1);

% 二次规划求解
options = optimoptions('quadprog', 'Display', 'off');
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);

% 计算测试集的核函数
for i = 1:size(test, 1)
    for j = 1:size(train, 1)
        dist = sum(nam * (Xmid1(i, :) - Xmid(j, :)).^2 + ...
            (1 - nam) * (Xwid1(i, :) - Xwid(j, :)).^2);
        K1(i, j) = exp(-dist / (2 * sigma^2));
    end
end

% 预测
predict = sign(K1 * alpha - sum(alpha .* diag(K)));

% 计算分类正确率
accuracy = mean(predict == test_labels);
fprintf('总体分类正确率: %.2f%%\n', accuracy * 100);

% 计算正类和负类的分类正确率
positiveAccuracy = mean(predict(test_labels == 1) == test_labels(test_labels == 1)) * 100;
negativeAccuracy = mean(predict(test_labels == -1) == test_labels(test_labels == -1)) * 100;
fprintf('正类分类正确率: %.2f%%\n', positiveAccuracy);
fprintf('负类分类正确率: %.2f%%\n', negativeAccuracy);
toc
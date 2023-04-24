clear; clc; close all;

% 导入时间序列
raw = importdata("SH.xlsx");
seq = raw.data; % 时间序列的数据
dates = raw.textdata(2:end,1); % 时间序列的刻度

%%%%%%%%
% data = readtable("T10Y2Y.csv")
% dates = data.DATE
% rate = data.T10Y2Y 
% nanRows = any(isnan(rate),2)
% dates = dates(~nanRows,:)
% rate = rate(~nanRows,:)
% 
% seq = log(rate(2:end, 1)./rate(1:end-1,1))
% 状态预测
mode = 5;
s_pred = markov_switching_model(seq,mode,'trend');
% % 绘图：左轴
% date = datetime(dates, 'InputFormat', 'yyyy-MM-dd');
% figure;
% hold on;
% title("The Result of MSH(2)-AR(1) Model",'FontSize',12)
% yyaxis left
% plot(date,seq,"m");
% set(gca,'ycolor',"m");
% ylabel("上证180指数");
% % 绘图：右轴
% yyaxis right
% plot(date,s_pred,'color',[0 0 0]);
% set(gca,'ycolor',[0 0 0]);
% ylabel("位于区制0的概率");

% 绘图：左轴
date = datetime(dates, 'InputFormat', 'yyyy-MM-dd');
figure;
hold on;
title("MSH(2)-AR(1)区制0平滑概率估计结果",'FontSize',12)
plot(date,s_pred,'color',[0 0 0]);
set(gca,'ycolor',[0 0 0]);
%ylabel("位于区制0的概率");

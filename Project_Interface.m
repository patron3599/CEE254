%% CEE 254 Project
% Team 2
% 11/1/2021

%% Clear cache
clear all
close all
clc

%% Define plotting defaults
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on',...
    'DefaultAxesXminortick','on','DefaultAxesYminortick','on',...
    'DefaultLineLineWidth',2,'DefaultLineMarkerSize',6,...
    'DefaultAxesFontName','Arial','DefaultAxesFontSize',14,...
    'DefaultAxesFontWeight','bold',...
    'DefaultTextFontWeight','normal','DefaultTextFontSize',14)

% input problem
problem_name = 'short_term';
problem_type = 1;
var_level = 5;
train_data = load(['train_data_',problem_name,'_',num2str(var_level),'_var.mat']).train_data; 
test_data = load(['test_data_',problem_name,'_',num2str(var_level),'_var.mat']).test_data; 

pred_pm2d5 = pm2d5_pred_model(train_data, test_data, problem_type);

save([problem_name,'_',num2str(var_level),'.mat'],'pred_pm2d5');
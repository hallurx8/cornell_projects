%% ---------------------------------------------------
% 马尔可夫区制转移模型总入口函数
% 模型形式：
% y_t - miu_s_t = beta_s_t + alpha_s_t * (y_t_1 - min_s_t_1) + e_t
% e_t~N(0,sigma_s_t)
% s_t为状态变量，取0或1（如果是0和1之间的实数，其含义为状态为1的概率）
% Q为s_t的状态转移矩阵
% 参数说明：
% seq: 待建模的时间序列
% mode: 整数，采用的模型变体编号（未来将进一步引入更多变体）
% 1. MSMH(2)-AR(0)
% 2. MSI(2)-AR(1)
% 3. MSM(2)-AR(1)
% 4. MSA(2)-AR(1)
% 5. MSH(2)-AR(1)
% 6. MSAI(2)-AR(1)
% init: 字符串，状态序列s的初始化方法：
% 'trend': 上行(1)或下行(0)趋势
% 'vol': 高波动(1)或低波动(0)状态
% 输出说明：
% s_pred: 预测t+1期状态为1（因子上行、资产高波动等）的概率
% ---------------------------------------------------
function s_pred = markov_switching_model(seq, mode, init)

% 时间序列去除缺失值
y = seq(~isnan(seq));

%% ---------------------------------------------------
% 状态序列s的初始化
% ---------------------------------------------------
s0 = nan(length(y),1);

switch init
    case 'trend' % 采用因子动量法初始化
        for t = 3:length(y)
            if (y(t) - y(t-1) > 0) && (y(t-1) - y(t-2) > 0)
                s0(t) = 1; % 连续两期上行，初始化为上行
            elseif (y(t) - y(t-1) < 0) && (y(t-1) - y(t-2) < 0)
                s0(t) = 0; % 连续两期下行，初始化为下行
            else
                s0(t) = 0.5; % 其余情况初始化为无法给出判断
            end
        end
        
        % 初始两期无法给出观点
        s0(1) = 0.5;
        s0(2) = 0.5;
    
    case 'vol' % 采用布林带初始化
        y_mean = mean(y);
        y_std = mean(y);
        
        for t = 1:length(y)
            if abs(y(t) - y_mean) < y_std * 1.5
                s0(t) = 0; % 在1.5倍标准差以内，初始化为低波动
            elseif abs(y(t) - y_mean) > y_std * 2.5
                s0(t) = 1; % 在2.5倍标准差以外，初始化为高波动
            else
                s0(t) = 0.5; % 其余情况初始化为无法给出判断
            end
        end
end

%% ---------------------------------------------------
% 模型求解
% ---------------------------------------------------
switch mode
    case 1 % MSMH(2)-AR(0)
        s_pred = MSMH2_AR0(y,s0);
    case 2 % MSI(2)-AR(1)
        s_pred = MSI2_AR1(y,s0);
    case 3 % MSM(2)-AR(1)
        s_pred = MSM2_AR1(y,s0);
    case 4 % MSA(2)-AR(1)
        s_pred = MSA2_AR1(y,s0);
    case 5 % MSH(2)-AR(1)
        s_pred = MSH2_AR1(y,s0);
    case 6 % MSAI(2)-AR(1)
        s_pred = MSAI2_AR1(y,s0);
end

end



%% ---------------------------------------------------
% 变体编号1
% MSMH(2)-AR(0)模型：alpha_s_t=0，beta_s_t=0，miu_s_t和sigma_s_t因s_t而异
% 参数说明（其他变体相同，不再赘述）：
% y: 待建模的时间序列，列向量；s0: 初始化后的状态序列，列向量
% 输出说明：
% s_pred: t时刻状态的预测概率，即P(s_t+1|y_t,y_t-1,...,y_1,y_0)
% ---------------------------------------------------
function s1 = MSMH2_AR0(y,s0)

% 时间序列长度（由于有0期状态，比待建模的截面多1）
T = length(y);

% 根据MSMH(2)-AR(0)模型的定义，alpha和beta强制为0
alpha = 0;
beta = 0;

% 通用模型（用于计算似然函数）
F = @(y_1, y_0, u_1, u_0, sigma, alpha, beta)...
     (1 / (sqrt(2 * pi) * sigma)) * exp(-((y_1 - u_1 - beta - alpha * (y_0 - u_0))^2) / (2 * sigma^2));

% 初始化EM算法起点
count = 0; % 迭代次数
last_L = -999999; % 上一次迭代后的似然函数

while count < 100
    
    %% ---------------------------------------------------
    % Maximization步
    % ---------------------------------------------------
    % 引入一些常量，便于反复调用
    A = sum(1 - s0(2:end));
    B = sum(s0(2:end));
    
    u_0 = dot(1 - s0(2:end), y(2:end)) / A;
    u_1 = dot(s0(2:end), y(2:end)) / B;
    
    sigma_0 = sqrt(dot(1 - s0(2:end), (y(2:end) - u_0).^2) / A);
    sigma_1 = sqrt(dot(s0(2:end), (y(2:end) - u_1).^2) / B);
    
    p_00 = dot(1 - s0(2:end), 1 - s0(1:end-1)) / sum(1 - s0(1:end-1));
    p_11 = dot(s0(2:end), s0(1:end-1)) / sum(s0(1:end-1));
    Q = [p_00, 1 - p_11;1 - p_00, p_11];

    %% ---------------------------------------------------
    % 计算模型似然值
    % ---------------------------------------------------
    % 使用已知参数代入通用函数
    f00 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_0, u_0, sigma_0, alpha, beta);
    f01 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_0, u_1, sigma_0, alpha, beta);
    f10 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_1, u_0, sigma_1, alpha, beta);
    f11 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_1, u_1, sigma_1, alpha, beta);
    
    % 初始化滤波概率gamma_00和预测概率gamma_10
    gamma_00 = zeros(2,T);
    gamma_10 = zeros(2,T);
    
    % t=0时刻（对应索引为1，其他类似）滤波概率更新
    gamma_00(:,1) = [1 - s0(1), s0(1)]';
    
    % 初始化似然函数
    cur_L = 0;
    
    for t = 2:T
        
        % 迭代计算预测概率
        gamma_10(:,t) = Q * gamma_00(:,t-1);
        
        % 计算状态条件概率
        P_s0 = [f00(y(t),y(t-1),alpha,beta), f01(y(t),y(t-1),alpha,beta)] * gamma_00(:,t-1);
        P_s1 = [f10(y(t),y(t-1),alpha,beta), f11(y(t),y(t-1),alpha,beta)] * gamma_00(:,t-1);
        
        % 计算当前时刻似然函数
        C = [P_s0, P_s1] * gamma_10(:,t);
        cur_L = cur_L + log(C);
        
        % 迭代计算滤波概率
        gamma_00(:,t) = [P_s0 / C; P_s1 / C] .* gamma_10(:,t);
        
    end
    if count == 1
        disp(gamma_00(1,:));
        disp(gamma_10(1,:));
    end

    
    % 判断似然函数是否收敛，若收敛则跳出循环
    if abs(cur_L - last_L) < 1.0e-6
        break
    else
        last_L = cur_L; 
        count = count + 1;
    end
    
    %% ---------------------------------------------------
    % Expectation步
    % ---------------------------------------------------
    % 初始化平滑概率gamma_tT
    gamma_tT = zeros(2,T);
    gamma_tT(:,T) = gamma_00(:,T);
    
    for t = T-1:-1:1
        gamma_tT(:,t) = (Q' * (gamma_tT(:,t+1) ./ gamma_10(:,t+1))) .* gamma_00(:,t);
    end
    
    % 用平滑概率计算状态的期望值，更新状态估计
    s0 = gamma_tT(2,:)';
    
end

% 计算未来1期的预测概率
s_pred = Q * gamma_00(:,T);

% 输出预测概率序列
s_pred = [gamma_00(2,2:end)';s_pred(2)];
s1 = s0;
format short g; 
fprintf("%8.4f\n",sigma_0);
fprintf("%8.4f\n",sigma_1);
fprintf("%8.4f\n",p_00);
fprintf("%8.4f\n",p_11);
fprintf("%8.4f\n",1/(1-p_00));
fprintf("%8.4f\n",1/(1-p_11));
fprintf("%8.4f\n",cur_L);


end






%% ---------------------------------------------------
% 变体编号5
function s1 = MSH2_AR1(y,s0)

% 时间序列长度（由于有0期状态，比待建模的截面多1）
T = length(y);

% 根据MSMH(2)-AR(0)模型的定义，alpha和beta强制为0
u_0 = 0;
u_1 = 0;
beta = 0;

% 通用模型（用于计算似然函数）
F = @(y_1, y_0, u_1, u_0, sigma, alpha, beta)...
     (1 / (sqrt(2 * pi) * sigma)) * exp(-((y_1 - u_1 - beta - alpha * (y_0 - u_0))^2) / (2 * sigma^2));

% 初始化EM算法起点
count = 0; % 迭代次数
last_L = -999999; % 上一次迭代后的似然函数

while count < 100
    
    %% ---------------------------------------------------
    % Maximization步
    % ---------------------------------------------------
    % 引入一些常量，便于反复调用
    A = sum(y(2:end).^2);
    B = sum(y(1:end-1).^2);
    C = dot(y(1:end-1), y(2:end));
    D = dot(s0(2:end), y(2:end).^2);
    E = dot(s0(2:end), y(1:end-1).^2); 
    FF = dot(s0(2:end), y(2:end) .* y(1:end-1));
    G = sum(s0(2:end));
    TT = T-1;
    c3 = -TT * E * (B-E);
    c2 = (C*E-B*FF)*G + TT*(2*B*FF+C*E-3*E*FF);
    c1 = -(TT*D*(B-E) + 2*TT*FF*(C-FF) + (A*E-B*D)*G);
    c0 = TT*D*(C-FF) + (A*FF-C*D)*G;
    X = roots([c3 c2 c1 c0]);

    alpha = X(imag(X)==0);
    %disp(size(1 - s0(2:end)));
    %disp(size((y(2:end) - alpha * y(1:end-1)).^2));
    sigma_0 = sqrt(dot(1 - s0(2:end), (y(2:end) - alpha * y(1:end-1)).^2) / sum(1 - s0(2:end)));
    sigma_1 = sqrt(dot(s0(2:end), (y(2:end) - alpha * y(1:end-1)).^2) / sum(s0(2:end)));
    
    p_00 = dot(1 - s0(2:end), 1 - s0(1:end-1)) / sum(1 - s0(1:end-1));
    p_11 = dot(s0(2:end), s0(1:end-1)) / sum(s0(1:end-1));
    Q = [p_00, 1 - p_11;1 - p_00, p_11];

    %% ---------------------------------------------------
    % 计算模型似然值
    % ---------------------------------------------------
    % 使用已知参数代入通用函数
    f00 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_0, u_0, sigma_0, alpha, beta);
    f01 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_0, u_1, sigma_0, alpha, beta);
    f10 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_1, u_0, sigma_1, alpha, beta);
    f11 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_1, u_1, sigma_1, alpha, beta);
    
    % 初始化滤波概率gamma_00和预测概率gamma_10
    gamma_00 = zeros(2,T);
    gamma_10 = zeros(2,T);
    
    % t=0时刻（对应索引为1，其他类似）滤波概率更新
    gamma_00(:,1) = [1 - s0(1), s0(1)]';
    
    % 初始化似然函数
    cur_L = 0;
    
    for t = 2:T
        
        % 迭代计算预测概率
        gamma_10(:,t) = Q * gamma_00(:,t-1);
        
        % 计算状态条件概率
        P_s0 = [f00(y(t),y(t-1),alpha,beta), f01(y(t),y(t-1),alpha,beta)] * gamma_00(:,t-1);
        P_s1 = [f10(y(t),y(t-1),alpha,beta), f11(y(t),y(t-1),alpha,beta)] * gamma_00(:,t-1);
        
        % 计算当前时刻似然函数
        C = [P_s0, P_s1] * gamma_10(:,t);
        cur_L = cur_L + log(C);
        
        % 迭代计算滤波概率
        gamma_00(:,t) = [P_s0 / C; P_s1 / C] .* gamma_10(:,t);
        
    end


    
    % 判断似然函数是否收敛，若收敛则跳出循环
    if abs(cur_L - last_L) < 1.0e-6
        break
    else
        last_L = cur_L; 
        count = count + 1;
    end
    
    %% ---------------------------------------------------
    % Expectation步
    % ---------------------------------------------------
    % 初始化平滑概率gamma_tT
    gamma_tT = zeros(2,T);
    gamma_tT(:,T) = gamma_00(:,T);
    
    for t = T-1:-1:1
        gamma_tT(:,t) = (Q' * (gamma_tT(:,t+1) ./ gamma_10(:,t+1))) .* gamma_00(:,t);
    end
    
    % 用平滑概率计算状态的期望值，更新状态估计
    s0 = gamma_tT(2,:)';
    
end

% 计算未来1期的预测概率
s_pred = Q * gamma_00(:,T);

% 输出预测概率序列
s_pred = [gamma_00(2,2:end)';s_pred(2)];
s1 = s0;
format short g; 
fprintf("%8.4f\n",sigma_0);
fprintf("%8.4f\n",sigma_1);
fprintf("%8.4f\n",p_00);
fprintf("%8.4f\n",p_11);
fprintf("%8.4f\n",1/(1-p_00));
fprintf("%8.4f\n",1/(1-p_11));
fprintf("%8.4f\n",cur_L);
disp([sigma_0, sigma_1, p_00, p_11, 1/(1-p_00), 1/(1-p_11), cur_L]) ;

end














%% ---------------------------------------------------
% 变体编号6
% MSAI(2)-AR(1)模型：miu_s_t=0，sigma_s_t=sigma，alpha_s_t和beta_s_t因s_t而异
% ---------------------------------------------------
function s_pred = MSAI2_AR1(y,s0)

% 时间序列长度（由于有0期状态，比待建模的截面多1）
T = length(y);

% 根据MSAI(2)-AR(1)模型：miu_s_t=0，sigma_s_t=sigma，alpha_s_t和beta_s_t因s_t而异
u_0 = 0;
u_1 = 0;

% 通用模型（用于计算似然函数）
F = @(y_1, y_0, u_1, u_0, sigma, alpha, beta)...
     (1 / (sqrt(2 * pi) * sigma)) * exp(-((y_1 - u_1 - beta - alpha * (y_0 - u_0))^2) / (2 * sigma^2));

% 初始化EM算法起点
count = 0; % 迭代次数
last_L = -999999; % 上一次迭代后的似然函数

while count < 100
    
    %% ---------------------------------------------------
    % Maximization步
    % ---------------------------------------------------
    % 引入一些常量，便于反复调用
    A = [dot(1 - s0(2:end), y(1:end-1)), sum(1 - s0(2:end)); ...
        dot(1 - s0(2:end), y(1:end-1).^2), dot(1 - s0(2:end), y(1:end-1))];
    B = [dot(s0(2:end), y(1:end-1)), sum(s0(2:end)); ...
        dot(s0(2:end), y(1:end-1).^2), dot(s0(2:end), y(1:end-1))];
    C = [dot(1 - s0(2:end), y(2:end)); dot(1 - s0(2:end), y(2:end) .* y(1:end-1))];
    D = [dot(s0(2:end), y(2:end)); dot(s0(2:end), y(2:end) .* y(1:end-1))];
    solution_0 = A \ C;
    alpha_0 = solution_0(1);
    beta_0 = solution_0(2);
    solution_1 = B \ D;
    alpha_1 = solution_1(1);
    beta_1 = solution_1(2);
    sigma = sqrt((dot(1 - s0(2:end), (y(2:end) - beta_0 - alpha_0 * y(1:end-1)).^2)...
        + dot(s0(2:end), (y(2:end) - beta_1 - alpha_1 * y(1:end-1)).^2) ) / (T - 1)); 
    p_00 = dot(1 - s0(2:end), 1 - s0(1:end-1)) / sum(1 - s0(1:end-1));
    p_11 = dot(s0(2:end), s0(1:end-1)) / sum(s0(1:end-1));
    Q = [p_00, 1 - p_11;1 - p_00, p_11];

    %% ---------------------------------------------------
    % 计算模型似然值
    % ---------------------------------------------------
    % 使用已知参数代入通用函数
    f00 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_0, u_0, sigma, alpha, beta);
    f01 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_0, u_1, sigma, alpha, beta);
    f10 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_1, u_0, sigma, alpha, beta);
    f11 = @(y_1, y_0, alpha, beta)F(y_1, y_0, u_1, u_1, sigma, alpha, beta);
    
    % 初始化滤波概率gamma_00和预测概率gamma_10
    gamma_00 = zeros(2,T);
    gamma_10 = zeros(2,T);
    
    % t=0时刻（对应索引为1，其他类似）滤波概率更新
    gamma_00(:,1) = [1 - s0(1), s0(1)]';
    
    % 初始化似然函数
    cur_L = 0;
    
    for t = 2:T
        
        % 迭代计算预测概率
        gamma_10(:,t) = Q * gamma_00(:,t-1);
        
        % 计算状态条件概率
        P_s0 = [f00(y(t),y(t-1),alpha_0,beta_0), f01(y(t),y(t-1),alpha_0,beta_0)] * gamma_00(:,t-1);
        P_s1 = [f10(y(t),y(t-1),alpha_1,beta_1), f11(y(t),y(t-1),alpha_1,beta_1)] * gamma_00(:,t-1);
        
        % 计算当前时刻似然函数
        C = [P_s0, P_s1] * gamma_10(:,t);
        cur_L = cur_L + log(C);
        
        % 迭代计算滤波概率
        gamma_00(:,t) = [P_s0 / C; P_s1 / C] .* gamma_10(:,t);
        
    end
    
    % 判断似然函数是否收敛，若收敛则跳出循环
    if abs(cur_L - last_L) < 1.0e-6
        break
    else
        last_L = cur_L; 
        count = count + 1;
    end
    
    %% ---------------------------------------------------
    % Expectation步
    % ---------------------------------------------------
    % 初始化平滑概率gamma_tT
    gamma_tT = zeros(2,T);
    gamma_tT(:,T) = gamma_00(:,T);
    
    for t = T-1:-1:1
        gamma_tT(:,t) = (Q' * (gamma_tT(:,t+1) ./ gamma_10(:,t+1))) .* gamma_00(:,t);
    end
    
    % 用平滑概率计算状态的期望值，更新状态估计
    s0 = gamma_tT(2,:)';
    
end

% 计算未来1期的预测概率
s_pred = Q * gamma_00(:,T);

% 输出预测概率序列
s_pred = [gamma_00(2,2:end)';s_pred(2)];

end








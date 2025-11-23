clc; clear; close all;

%% Load Data
load('pi0.mat');   % pi0 should be numeric vector

%% Parameters
gamma = -1.5;
E0 = 25;
E_min = 50;
E_max = 500;
hours = 24;

eff_c = 0.9;
eff_d = 0.9;
eff_s = 0.99;

%% Decision Variables
n = 2*hours;   % [x1..xT (charge); x(T+1)..x(2T) (discharge)]

%% Symbolic Revenue (for H, f)
syms x [n 1] 
diff = x(hours+1:end) - x(1:hours);      % P_dis - P_chg
pi_sym = pi0(1:hours).' + gamma*diff;    % Day-ahead price
Revenue = - pi_sym.' * diff;             % Negative for minimization

% Compute H and f for quadprog
f = double(subs(gradient(Revenue,x), x, zeros(n,1)));
H = 0.5 * double(hessian(Revenue,x));

%% Energy Constraints
A = [];
b = [];

for t = 1:hours
    row = zeros(1,n);
    for k = 1:t
        row(k) = eff_c * eff_s^(t-k);         % charging coeff
        row(hours+k) = -1/eff_d * eff_s^(t-k); % discharging coeff
    end
    % Upper bound
    A = [A; row];
    b = [b; E_max - E0];
    
    % Lower bound
    A = [A; -row];
    b = [b; -(E_min - E0)];
end

%% Equality constraint (E_T = E0)
Aeq = zeros(1,n);
for k = 1:hours
    Aeq(1,k) = eff_c * eff_s^(hours - k);
    Aeq(1,hours + k) = -1/eff_d * eff_s^(hours - k);
end
beq = 0;

%% Variable bounds
lb = 0* ones(n,1);
ub = 100 * ones(n,1);

%% Solve Quadratic Program
opts = optimset('Algorithm','interior-point-convex','Display','iter');
[x_opt, fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],opts);

%% Display Results
disp('Optimal Charging/Discharging Power:');
disp(x_opt);

disp('Maximum Revenue:');
disp(-fval);  % Flip sign because we minimized -Revenue

%%Plot for Energy
%% Compute Energy Level over Time
E = zeros(hours+1,1);     % Energy storage for each hour
E(1) = E0;                % Initial energy

for t = 1:hours
    % charging and discharging power
    P_chg = x_opt(t);
    P_dis = x_opt(hours + t);
    
    % update energy
    E(t+1) = eff_s * E(t) + eff_c * P_chg - (1/eff_d) * P_dis;
end

%% Plot Results (Smooth Curves)
time_E = 0:hours;            % 25 points for energy (E0...E24)
time_P = 1:hours;            % 24 points for charging/discharging

% Smooth the data (moving average)
E_smooth     = smoothdata(E, 'movmean', 3);
P_chg_smooth = smoothdata(x_opt(1:hours), 'movmean', 3);
P_dis_smooth = smoothdata(-x_opt(hours+1:end), 'movmean', 3);  % negative for clarity

%% Create Figure
figure('Units','centimeters','Position',[3 3 18 10]);  % for consistent sizing
set(gcf,'Color','w');  % white background

% Left y-axis (Energy)
yyaxis left
plot(time_E, E_smooth, 'LineWidth', 2.2, 'Color', [0 0.447 0.741]);
ylabel('Energy Level (MWh)', 'FontSize', 11, 'FontWeight', 'bold');
ylim([E_min E_max]);

% Right y-axis (Power)
yyaxis right
plot(time_P, P_chg_smooth, '--', 'LineWidth', 2, 'Color', [0.466 0.674 0.188]); hold on;
plot(time_P, P_dis_smooth, '--', 'LineWidth', 2, 'Color', [0.85 0.325 0.098]);
ylabel('Power (MW)', 'FontSize', 11, 'FontWeight', 'bold');

% Labels and title
xlabel('Time (hours)', 'FontSize', 11, 'FontWeight', 'bold');
title('Smooth Energy, Charging, and Discharging Profile', 'FontSize', 12, 'FontWeight', 'bold');

% Legend and grid
legend({'Energy Level','Charging Power','Discharging Power'}, 'Location','best', 'FontSize', 9);
grid on;
xlim([0 hours]);
ax = gca;
ax.FontSize = 10;
ax.LineWidth = 1;

%% Export as SVG (for LaTeX or publication)
print(gcf, 'smooth_profile', '-dsvg', '-r300');   % exports as smooth_profile.svg




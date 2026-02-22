clc
clear
%% 参数设置                                                                                                                  
%% Load 3D data
% load '.\data1.mat'
tic
load 'G:\matlab\3d\p4\simulation\sim_1\data_noise\cscan_data_s_noise_4000\GLZO_gan1_noise4000.mat'
load 'data_void.mat'

% data_3d维度: P x M x Nx, where P is time samples, M is B-scans in y, Nx is traces in x
% 数据预处理
data_3d = data_3d - data_3d_void - data_3d_void;
data_3d(401:end, :, :) = [];  % 截取前400个时间采样点

% 获取数据维度
P = size(data_3d, 1);  % 时间采样点数
M = size(data_3d, 2);   % y方向的B-Scan数
Nx = size(data_3d, 3);  % x方向的轨迹数（traces）

% 对每个B-Scan（第二维M）进行均值滤波
for ii = 1:M   % 遍历所有B-Scan
    data_3d(:, ii, :) = data_3d(:, ii, :) - mean(data_3d(:, ii, :), 'all');
end

% Pad to reduce spectrum fence effect
pad_size = 20001 - P;
if pad_size > 0
    data_3d = [data_3d; zeros(pad_size, M, Nx)];
end
dt = 6.1e-12;

% 更新P为填充后的时间采样点数
P = size(data_3d, 1);

%% Parameters
Epsilon = 3.5;
C = 3e8 / sqrt(Epsilon);
f0 = 5e9;
Lamda = C / f0;
t_delay = 2.8 / f0;  % Time delay for zero correction
DX = 0.005;  % Trace spacing in x direction
DY = 0.01;   % Trace spacing in y direction
Zrange = 0.10;  % Z imaging scene size
start_step = 0;

% 根据实际天线位置计算成像范围
% 天线实际覆盖范围：x方向 (Nx-1)*DX，y方向 (M-1)*DY
% 成像范围应该匹配天线覆盖范围，或者根据实际需求设置
Xrange_actual = (Nx - 1) * DX;  % x方向实际天线覆盖范围
Yrange_actual = (M - 1) * DY;    % y方向实际天线覆盖范围

% 成像场景大小（可以根据需要调整，但应该与天线覆盖范围匹配）
Xrange = Xrange_actual;  % 使用实际天线覆盖范围，或者设置为期望的成像范围
Yrange = Yrange_actual;  % 使用实际天线覆盖范围，或者设置为期望的成像范围

% 显示信息
fprintf('天线覆盖范围: x方向=%.4f m (Nx=%d, DX=%.4f), y方向=%.4f m (M=%d, DY=%.4f)\n', ...
    Xrange_actual, Nx, DX, Yrange_actual, M, DY);
fprintf('成像网格范围: x方向=%.4f m, y方向=%.4f m\n', Xrange, Yrange);

%% 天线网格位置
% 根据参考代码，天线位置应该是 [M, Nx] 矩阵
% 注意：data_3d维度是 P x M x Nx
% - M: y方向的B-Scan数，对应参考代码中的M
% - Nx: x方向的轨迹数（traces），对应参考代码中的N
% x_line: x方向位置矩阵 [M, Nx] - 每一行对应一个B-Scan，每一列对应一个轨迹
% y_line_s: y方向发射位置矩阵 [M, Nx]
% y_line_r: y方向接收位置矩阵 [M, Nx]
% z_line: z方向位置矩阵 [M, Nx]

% 构建天线位置矩阵
% 参考代码中：x_line=(0.03:0.005:0.47).'*ones(1,N) - 每一行相同，对应x方向
%            y_line_s=ones(M,1)*(0.03:0.005:0.47) - 每一列相同，对应y方向
x_line = zeros(M, Nx);
y_line_s = zeros(M, Nx);
y_line_r = zeros(M, Nx);
z_line = zeros(M, Nx);

% 根据参考代码，x方向对应轨迹（Nx），y方向对应B-Scan（M）
% x_line: 每一行相同，x坐标沿列方向变化（对应轨迹索引i2）
% y_line_s: 每一列相同，y坐标沿行方向变化（对应B-Scan索引i1）
for i1 = 1:M  % B-Scan索引（y方向）
    for i2 = 1:Nx  % 轨迹索引（x方向）
        % x方向：沿轨迹方向（列方向），对应参考代码中的x_line
        x_line(i1, i2) = start_step + (i2 - 1) * DX;
        % y方向：沿B-Scan方向（行方向），对应参考代码中的y_line_s
        y_line_s(i1, i2) = start_step + (i1 - 1) * DY;
        y_line_r(i1, i2) = start_step + (i1 - 1) * DY + 0.005;  % 接收位置稍微偏移
        z_line(i1, i2) = 0.1;  % 天线高度
    end
end

%% 成像网格设置
% 确保成像网格是128x128x128
nx_grid = 128;
ny_grid = 128;
nz_grid = 128;

dx = Xrange / nx_grid;
dy = Yrange / ny_grid;
dz = Zrange / nz_grid;

% 使用linspace确保精确的点数
xAxis = linspace(0, Xrange, nx_grid);  % x方向成像区间（对应轨迹方向Nx），128个点
yAxis = linspace(0, Yrange, ny_grid);  % y方向成像区间（对应B-Scan方向M），128个点
fSceneZ_array = linspace(0, Zrange, nz_grid);  % z方向成像区间，128个点

% 验证点数
fprintf('成像网格点数: x方向=%d, y方向=%d, z方向=%d\n', ...
    length(xAxis), length(yAxis), length(fSceneZ_array));

% 构建成像网格矩阵
% 注意：如果发现x和y反了，可以交换xMatrix和yMatrix的定义
% xMatrix: 每一行相同，x坐标沿列方向变化 [ny, nx]
% yMatrix: 每一列相同，y坐标沿行方向变化 [ny, nx]
% 参考代码：x_line每一行相同（x沿列变化），y_line_s每一列相同（y沿行变化）
xMatrix = ones(length(yAxis), 1) * xAxis;  % [ny, nx] - 每一行是xAxis，x沿列方向变化
yMatrix = yAxis' * ones(1, length(xAxis));  % [ny, nx] - 每一列是yAxis，y沿行方向变化

% 如果x和y反了，取消下面的注释来交换
% temp = xMatrix;
% xMatrix = yMatrix;
% yMatrix = temp;

% 成像深度序列已在上面定义
nSceneZ = length(fSceneZ_array);  % 成像场景深度个数

%% Initialize image
KircImage = zeros(length(yAxis), length(xAxis), nSceneZ);

%% 数据平面法线方向
vector_S = [0, 0, -1];

%% Kirchhoff migration
% 
for deep = 1:nSceneZ
    zMatrix = fSceneZ_array(deep) * ones(length(yAxis), length(xAxis));
    
    for i1 = 1:M  % 遍历B-Scan (y方向)
        for i2 = 1:Nx  % 遍历轨迹 (x方向)
            %% 向量从天线到成像点
            vector_X = xMatrix - x_line(i1, i2);
            vector_Y_s = yMatrix - y_line_s(i1, i2);
            vector_Y_r = yMatrix - y_line_r(i1, i2);
            vector_Z = zMatrix - z_line(i1, i2);
            
            %% 距离矩阵
            rangeMatrix_s = sqrt(vector_X.^2 + vector_Y_s.^2 + vector_Z.^2);
            rangeMatrix_r = sqrt(vector_X.^2 + vector_Y_r.^2 + vector_Z.^2);
            
            % 避免除零
            [row, column] = find(rangeMatrix_s == 0);
            for j = 1:length(row)
                rangeMatrix_s(row(j), column(j)) = eps;
            end
            
            %% 两个向量之间的夹角余弦值
            % 使用平均距离计算cos_theta
            cos_theta = (vector_X * vector_S(1) + vector_Y_s * vector_S(2) + vector_Z * vector_S(3)) ./ ...
                       ((rangeMatrix_s + rangeMatrix_r) / 2) / sqrt(vector_S * vector_S.');
            
            %% 信号和导数
            % data_3d维度: P x M x Nx
            % 提取第i1个B-Scan、第i2个x轨迹的所有时间采样点
            signalTemp = squeeze(data_3d(:, i1, i2));  % [P, 1]
            signalTemp1 = diff(signalTemp) / dt;
            signalTemp1 = [signalTemp1; signalTemp1(end)];  % Pad last value
            
            %% 时间索引
            % 使用往返距离计算时间索引
            nRangeCell = round((rangeMatrix_r + rangeMatrix_s) / (C * dt));
            nRangeCell(nRangeCell < 1) = 1;
            nRangeCell(nRangeCell > P - 1) = P - 1;
            
            %% 累积贡献
            % 按照参考代码的公式
            contrib = (cos_theta ./ rangeMatrix_s / C) .* signalTemp1(nRangeCell) + ...
                     (cos_theta ./ (rangeMatrix_r.^2)) .* signalTemp(nRangeCell);
            
            KircImage(:, :, deep) = KircImage(:, :, deep) + contrib;
        end
    end
    
    % 显示进度
    if mod(deep, 10) == 0
        fprintf('处理深度 %d/%d\n', deep, nSceneZ);
    end
end

%% Normalize
KircImage = abs(KircImage) / max(abs(KircImage(:)));
KircImage(isnan(KircImage)) = 0;

% 保存结果
save('test_kirhf_Images1.mat', 'KircImage');
toc

%% Display (example, slice or 3D visualization)
load test_kirhf_Images1.mat
KircImage(KircImage < 0.35) = NaN;

% 验证图像尺寸
fprintf('KircImage尺寸: %d x %d x %d\n', size(KircImage, 1), size(KircImage, 2), size(KircImage, 3));

% 使用与成像网格相同的坐标
%% 改成 ndgrid 格式 —— 这才是 slice 真正需要的！
[y,x,z] = ndgrid(dy:dy:0.40, dx:dx:0.28, dz:dz:0.10);
% 注意顺序：第一个输出是 y，第二个是 x！

% 下面这三行保持不变（或者直接用向量也行）
xs = dx:dx:0.28;
ys = dy:dy:0.40; 
zs = dz:dz:0.10;

figure('Position', [100, 100, 800, 700]);
h = slice(x, y, z, KircImage, xs, ys, zs);   % 注意这里 x,y,z 顺序不能乱！
% xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)');
axis([0,0.28,0,0.40,0,0.10]);
set(h,'FaceColor','interp','EdgeColor','none');
set(gca,'linewidth',1,'fontsize',16,'fontname','Times New Roman');
set(gca,'ydir','reverse');
set(gca,'zdir','reverse');
camproj perspective;box on;
view(54,27);
colormap ('jet');colorbar();
axis equal;grid on;

% 设置正确的宽高比，确保x和y方向的比例正确
daspect([Xrange Yrange Zrange]);  % 根据实际物理尺寸设置宽高比
grid on;

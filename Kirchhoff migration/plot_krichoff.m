clc
clear

load kirchhoff_image.mat
KircImage = migration_volume;
KircImage(KircImage < 0.35) = NaN;

Xrange = 0.28;  % 
Yrange = 0.40;  % 
Zrange = 0.10;  % 


%% 成像网格设置
% 确保成像网格是128x128x128
nx_grid = 128;
ny_grid = 128;
nz_grid = 128;

dx = Xrange / nx_grid;
dy = Yrange / ny_grid;
dz = Zrange / nz_grid;

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
axis([0, Xrange, 0, Yrange, 0, Zrange]);

% 按物理尺寸设置绘图框比例；默认 pbaspect [1 1 1] 会显示为正方体
pbaspect([Xrange Yrange Zrange]);
daspect([1 1 1]);   % 保持数据单位等比例，由 pbaspect 控制整体形状
grid on;

%% Save figure
fig_save_path = 'kirchhoff_slice.png';       % Output path (change as needed)
print(gcf, fig_save_path, '-dpng', '-r300'); % PNG, 300 dpi
% saveas(gcf, 'kirchhoff_slice.fig');         % Uncomment for .fig format
% print(gcf, 'kirchhoff_slice.pdf', '-dpdf');% Uncomment for PDF (vector)
fprintf('Figure saved: %s\n', fig_save_path);

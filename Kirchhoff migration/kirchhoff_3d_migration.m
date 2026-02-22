clc
clear

%% =========================================================================
%  3D GPR Kirchhoff Migration
%  Bistatic geometry with obliquity factor; zero-phase Ricker wavelet assumed.
%
%  Usage:
%    1. Set data_path below to your .mat file path (e.g. 'C:/data/gpr_survey.mat')
%    2. Ensure the .mat contains variable 'data_3d' with size [time x bscans x traces]
%    3. Run this script
%% =========================================================================

%% Configuration (primary inputs)
data_path = '/workspace/MY_SSW/SSW/p4/mvnet/baselines/krichof_t/data_example/GLZO_gan1_noise4000.mat';                              % e.g. '/path/to/your/data.mat'
data_var_name = 'data_3d';                   % Variable name in .mat file

% Propagation velocity in medium (m/s). Permittivity derived as epsilon_r = (c0/v)^2.
c0 = 3e8;                                    % Speed of light in vacuum (m/s)
medium_velocity = 1.6e8;                     % velocity
epsilon_r = (c0 / medium_velocity)^2;        % Relative permittivity (derived)

dx_trace = 0.005;                            % Trace spacing along x (m)
dy_trace = 0.01;                             % Trace spacing along y (m)
z_range = 0.10;                              % Imaging depth extent (m)
start_step = 0;                              % Antenna grid origin offset (m)
dt = 6.1e-12;                                % Sampling interval (s)
f0 = 5e9;                                    % Center frequency (Hz)

%% Data Loading
if isempty(data_path)
    error('Set data_path to your .mat file path. Example: data_path = ''C:/data/gpr_survey.mat'';');
end

tic
loaded = load(data_path, data_var_name);
if ~isfield(loaded, data_var_name)
    error('Variable ''%s'' not found in %s', data_var_name, data_path);
end
radargram_volume = loaded.(data_var_name);

radargram_volume(401:end, :, :) = [];        % Truncate to 400 samples (depth limit)

num_time_samples = size(radargram_volume, 1);
num_b_scans = size(radargram_volume, 2);
num_traces_x = size(radargram_volume, 3);

x_range = (num_traces_x - 1) * dx_trace;
y_range = (num_b_scans - 1) * dy_trace;

fprintf('Loaded: %s\n', data_path);
fprintf('Aperture: x=%.4f m (n_traces=%d, dx=%.4f), y=%.4f m (n_bscans=%d, dy=%.4f)\n', ...
    x_range, num_traces_x, dx_trace, y_range, num_b_scans, dy_trace);

%% Preprocessing
% DC removal per B-scan
radargram_volume = radargram_volume - mean(radargram_volume, 1);

% Zero-pad for FFT (reduce spectral leakage)
pad_size = 20001 - num_time_samples;
if pad_size > 0
    radargram_volume = [radargram_volume; zeros(pad_size, num_b_scans, num_traces_x)];
end
num_time_samples = size(radargram_volume, 1);

%% Imaging Grid Setup
nx_grid = 128;
ny_grid = 128;
nz_grid = 128;

dx = x_range / nx_grid;
dy = y_range / ny_grid;
dz = z_range / nz_grid;

x_axis = linspace(0, x_range, nx_grid);
y_axis = linspace(0, y_range, ny_grid);
z_depth_array = linspace(0, z_range, nz_grid);

fprintf('Grid: nx=%d, ny=%d, nz=%d (dx=%.6f, dy=%.6f, dz=%.6f m)\n', ...
    nx_grid, ny_grid, nz_grid, dx, dy, dz);

% Image domain: x along columns, y along rows
x_matrix = ones(ny_grid, 1) * x_axis;
y_matrix = y_axis' * ones(1, nx_grid);

%% Antenna Grid Setup
% Positions [num_b_scans x num_traces_x]; x along trace, y along B-scan
[trace_idx_grid, bscan_idx_grid] = meshgrid(0:num_traces_x-1, 0:num_b_scans-1);

x_antenna = start_step + trace_idx_grid * dx_trace;
y_antenna_tx = start_step + bscan_idx_grid * dy_trace;
y_antenna_rx = y_antenna_tx + 0.005;         % Bistatic receiver offset (m)
z_antenna = 0.1 * ones(num_b_scans, num_traces_x);

%% Migration Kernel
% Kirchhoff migration; data plane normal for obliquity factor
data_plane_normal = [0, 0, -1];
migration_volume = zeros(ny_grid, nx_grid, nz_grid);

for depth_idx = 1:nz_grid
    z_matrix = z_depth_array(depth_idx) * ones(ny_grid, nx_grid);

    for bscan_idx = 1:num_b_scans
        for trace_idx = 1:num_traces_x
            % Vectors from antenna to image points
            vec_x = x_matrix - x_antenna(bscan_idx, trace_idx);
            vec_y_tx = y_matrix - y_antenna_tx(bscan_idx, trace_idx);
            vec_y_rx = y_matrix - y_antenna_rx(bscan_idx, trace_idx);
            vec_z = z_matrix - z_antenna(bscan_idx, trace_idx);

            % Two-way travel distances (Tx and Rx paths)
            range_tx = sqrt(vec_x.^2 + vec_y_tx.^2 + vec_z.^2);
            range_rx = sqrt(vec_x.^2 + vec_y_rx.^2 + vec_z.^2);
            range_tx(range_tx == 0) = eps;

            % Obliquity factor: cosine of angle between range vector and data plane normal
            range_avg = (range_tx + range_rx) / 2;
            cos_theta = (vec_x * data_plane_normal(1) + vec_y_tx * data_plane_normal(2) + ...
                         vec_z * data_plane_normal(3)) ./ range_avg / norm(data_plane_normal);

            % Trace and time derivative for Kirchhoff integral
            trace_signal = squeeze(radargram_volume(:, bscan_idx, trace_idx));
            trace_derivative = diff(trace_signal) / dt;
            trace_derivative = [trace_derivative; trace_derivative(end)];

            % Time-to-sample index (two-way travel time)
            n_range_cell = round((range_rx + range_tx) / (medium_velocity * dt));
            n_range_cell = max(1, min(n_range_cell, num_time_samples - 1));

            % Kirchhoff integral contribution
            contrib = (cos_theta ./ range_tx / medium_velocity) .* trace_derivative(n_range_cell) + ...
                      (cos_theta ./ (range_rx.^2)) .* trace_signal(n_range_cell);

            migration_volume(:, :, depth_idx) = migration_volume(:, :, depth_idx) + contrib;
        end
    end

    if mod(depth_idx, 10) == 0
        fprintf('Depth slice %d/%d\n', depth_idx, nz_grid);
    end
end

%% Post-processing
migration_volume = abs(migration_volume) / max(abs(migration_volume(:)));
migration_volume(isnan(migration_volume)) = 0;

[out_dir, ~, ~] = fileparts(data_path);
if isempty(out_dir), out_dir = pwd; end
out_path = fullfile(out_dir, 'kirchhoff_image.mat');
save(out_path, 'migration_volume');
toc

fprintf('Saved: %s\n', out_path);
fprintf('Output size: %d x %d x %d\n', size(migration_volume, 1), size(migration_volume, 2), size(migration_volume, 3));

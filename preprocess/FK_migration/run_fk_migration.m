clc
clear

%% =========================================================================
%  3D GPR F-K (Stolt) Migration - Driver Script
%
%  Usage:
%    1. Set data_path below
%    2. Ensure .mat contains variable 'data_3d' with size [time x traces x bscans]
%    3. Run this script
%% =========================================================================

%% Configuration
data_path = '/workspace/MY_SSW/SSW/p4/mvnet/baselines/krichof_t/data_example/GLZO_gan1_noise4000.mat'; ;                              % e.g. '/path/to/data1.mat'
data_var_name = 'data_3d';

c0 = 3e8;
medium_velocity = 1.6e8;                     % m/s; epsilon_r = (c0/v)^2

dx_trace = 0.005;
dy_trace = 0.01;
z_range = 0.10;
start_step = 0;
dt = 6.1e-12;

nx_grid = 128;
ny_grid = 128;
nz_grid = 128;

%% Data Loading
if isempty(data_path)
    error('Set data_path to your .mat file path.');
end

loaded = load(data_path, data_var_name);
if ~isfield(loaded, data_var_name)
    error('Variable ''%s'' not found in %s', data_var_name, data_path);
end
radargram_volume = loaded.(data_var_name);

%% Preprocessing

num_time_samples = size(radargram_volume, 1);
num_b_scans = size(radargram_volume, 2);
num_traces_x = size(radargram_volume, 3);

x_range = (num_traces_x - 1) * dx_trace;
y_range = (num_b_scans - 1) * dy_trace;

pad_size = 17001 - num_time_samples;
if pad_size > 0
    radargram_volume = [radargram_volume; zeros(pad_size, num_b_scans, num_traces_x)];
end

fprintf('Loaded: %s\n', data_path);
fprintf('Aperture: x=%.4f m (n_traces=%d), y=%.4f m (n_bscans=%d)\n', ...
    x_range, num_traces_x, y_range, num_b_scans);

%% Migration
params = struct('medium_velocity', medium_velocity, 'dt', dt, ...
    'dx_trace', dx_trace, 'dy_trace', dy_trace, ...
    'x_range', x_range, 'y_range', y_range, 'z_range', z_range, ...
    'start_step', start_step, 'nx_grid', nx_grid, 'ny_grid', ny_grid, 'nz_grid', nz_grid);

tic
migration_volume = fk_stolt_migration(radargram_volume, params);
elapsed = toc;

fprintf('Migration time: %.4f s\n', elapsed);

%% Save
[out_dir, ~, ~] = fileparts(data_path);
if isempty(out_dir), out_dir = pwd; end
out_path = fullfile(out_dir, 'fk_migration_image.mat');
save(out_path, 'migration_volume');
fprintf('Saved: %s\n', out_path);

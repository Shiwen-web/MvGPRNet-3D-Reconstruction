function migration_volume = fk_stolt_migration(radargram_volume, params)
%FK_STOLT_MIGRATION  3D GPR F-K (Stolt) migration, GPU-accelerated.
%
%   migration_volume = fk_stolt_migration(radargram_volume, params)
%
%   Inputs:
%     radargram_volume  - [time x traces x bscans], after permute and zero-pad
%     params            - struct with fields:
%       .medium_velocity  (m/s)
%       .dt               (s)
%       .dx_trace, .dy_trace (m)
%       .x_range, .y_range, .z_range (m)
%       .start_step       (m)
%       .nx_grid, .ny_grid, .nz_grid
%
%   Output:
%     migration_volume  - [ny x nx x nz], normalized

medium_velocity = params.medium_velocity;
dt = params.dt;
dx_trace = params.dx_trace;
dy_trace = params.dy_trace;
x_range = params.x_range;
y_range = params.y_range;
z_range = params.z_range;
start_step = params.start_step;
nx_grid = params.nx_grid;
ny_grid = params.ny_grid;
nz_grid = params.nz_grid;

num_b_scans = size(radargram_volume, 3);
num_traces_x = size(radargram_volume, 2);
sz1 = size(radargram_volume, 1);

dx = x_range / nx_grid;
dy = y_range / ny_grid;
dz = z_range / nz_grid;

z_depth_array = dz:dz:z_range;
n_depth_slices = length(z_depth_array);

% Wavenumber grids
omega_tem = 2 * pi / (sz1 * dt) * ...
    ([1:floor(sz1 / 2), (sz1 - (floor(sz1 / 2) + 1:sz1))]).' * ones(1, num_traces_x);
Kx_tem = 2 * pi / (num_traces_x * dx_trace) * ([1:floor(num_traces_x / 2), (num_traces_x - (floor(num_traces_x / 2) + 1:num_traces_x))]);
Ky_tem = reshape(2 * pi / (num_b_scans * dy_trace) * ([1:floor(num_b_scans / 2), (num_b_scans - (floor(num_b_scans / 2) + 1:num_b_scans))]), [1, 1, num_b_scans]);
Kz_tem = 2 * pi / z_range * (1:n_depth_slices);

% Antenna grid: columns=bscans (spacing dx_trace), rows=traces (spacing dy_trace), match fk_jiasu
[x_grid, y_grid] = meshgrid(start_step:dx_trace:(num_b_scans-1)*dx_trace + start_step, start_step:dy_trace:(num_traces_x-1)*dy_trace + start_step);
[xs_grid, ys_grid] = meshgrid(dx:dx:x_range, dy:dy:y_range);

% GPU precomputation
radargram_gpu = gpuArray(radargram_volume);
fft_base_gpu = fftn(radargram_gpu);

omega_gpu = gpuArray(repmat(omega_tem, 1, 1, num_b_scans));
Kx_gpu = gpuArray(repmat(repmat(Kx_tem, sz1, 1), 1, 1, num_b_scans));
Ky_gpu = gpuArray(repmat(Ky_tem, sz1, num_traces_x, 1));

% Stolt mapping
Kz_gpu = sqrt(complex(4 * omega_gpu.^2 / medium_velocity^2 - Kx_gpu.^2 - Ky_gpu.^2));
temp_gpu = 4 * omega_gpu.^2 / medium_velocity^2 - Kx_gpu.^2 - Ky_gpu.^2;
Kz_gpu(floor(size(Kz_gpu, 1) / 2):end, :, :) = -Kz_gpu(floor(size(Kz_gpu, 1) / 2):end, :, :);
Kz_gpu(1, :, :) = 0;

fft_mig_gpu = fft_base_gpu;
fft_mig_gpu(temp_gpu < 0) = 0;
fft_mig_gpu(4 * omega_gpu.^2 / medium_velocity^2 > ((pi / dx)^2 + (pi / dy)^2 + (pi / dz)^2)) = 0;
fft_mig_gpu = fft_mig_gpu .* exp(1i * Kz_gpu * z_range);

% Stolt interpolation
conv_fft_gpu = gpuArray(zeros(num_traces_x, num_b_scans, n_depth_slices));
for kx = 1:num_traces_x
    for ky = 1:num_b_scans
        index = floor(sz1 * dt * medium_velocity / (4 * pi) * ...
            sqrt((Kx_tem(1, kx))^2 + (Ky_tem(1, ky))^2 + Kz_tem.^2) + 1);
        index(((z_range/dz/2)+1):end) = -index(((z_range/dz)/2):-1:1) + 16384;
        mval = medium_velocity / 2 * Kz_tem ./ sqrt((Kx_tem(1, kx))^2 + (Ky_tem(1, ky))^2 + Kz_tem.^2 + eps) .* ...
            (gather(fft_mig_gpu(index, kx, ky))).';
        conv_fft_gpu(kx, ky, :) = gpuArray(reshape(mval, [1, 1, n_depth_slices]));
    end
end

res = gather(abs(ifftn(conv_fft_gpu)));

% Interpolate to imaging grid
migration_volume = zeros(y_range/dy, x_range/dx, z_range/dz);
for depth_idx = 1:n_depth_slices
    migration_volume(:, :, depth_idx) = interp2(x_grid, y_grid, res(:, :, depth_idx), xs_grid, ys_grid, 'cubic');
end
migration_volume = migration_volume / max(migration_volume(:));
migration_volume(isnan(migration_volume)) = 0;


end

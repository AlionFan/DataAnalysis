%% Optical DAT Viewer — MATLAB version
%  Read .dat file, compute radius-intensity curve, and plot.
%  Ported from the Python project optics_dat_viewer.

clear; clc; close all;

%% ==================== Configuration ====================
dat_path     = '';            % e.g. '/Users/fan/Downloads/guangyuan1000wan.dat'
center_row   = 50.5;         % center row index (1-based in MATLAB)
center_col   = 50.5;         % center col index (1-based in MATLAB)
max_radius   = 50;           % maximum radius (in pixel units)
dist_mode    = 'euclidean';  % 'euclidean' or 'manhattan'
offset_row   = 0;            % coordinate offset (subtracted before distance calc)
offset_col   = 0;

%% ==================== 1. Read .dat file ====================
[grid, info] = read_dat(dat_path);

fprintf('Loaded: %s\n', dat_path);
fprintf('  Grid size : %d x %d\n', info.rows, info.cols);
fprintf('  Hole value: %g\n', info.hole_value);
fprintf('  NaN pixels: %d / %d\n', sum(isnan(grid(:))), numel(grid));

%% ==================== 2. Compute radius-intensity curve ====================
[radii, means, counts] = radius_intensity_curve( ...
    grid, center_row, center_col, max_radius, dist_mode, offset_row, offset_col);

%% ==================== 3. Compute radius-mean (aggregate stats) ====================
r_test = 5;
rm = radius_mean(grid, center_row, center_col, r_test, dist_mode, offset_row, offset_col);
fprintf('\n--- Radius-mean (r=%.1f, %s) ---\n', r_test, dist_mode);
fprintf('  mean  = %.6g\n', rm.mean);
fprintf('  std   = %.6g\n', rm.std);
fprintf('  count = %d\n', rm.count);
fprintf('  min   = %.6g\n', rm.min);
fprintf('  max   = %.6g\n', rm.max);

%% ==================== 4. Plot R-Intensity curve ====================
figure('Name', 'Radius-Intensity Curve', 'Position', [100 100 900 500]);

% --- Manhattan subplot ---
[r_m, mean_m, cnt_m] = radius_intensity_curve( ...
    grid, center_row, center_col, max_radius, 'manhattan', offset_row, offset_col);

subplot(1, 2, 1);
plot(r_m, mean_m, '-o', 'MarkerSize', 3, 'LineWidth', 1.2);
xlabel('Radius (Manhattan distance)');
ylabel('Mean Intensity');
title('Manhattan Distance R-Intensity Curve');
grid on;

subplot(1, 2, 2);
% Scatter all valid pixels with their euclidean distance
[rr, cc] = ndgrid(1:size(grid,1), 1:size(grid,2));
dr = rr - offset_row - center_row;
dc = cc - offset_col - center_col;
euc_dist = sqrt(dr.^2 + dc.^2);
valid = ~isnan(grid);
scatter(euc_dist(valid), grid(valid), 1, 'o', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15, 'r');
hold on;
plot(radii, means, 'r-', 'LineWidth', 2);
xlabel('Radius (Euclidean distance)');
ylabel('Intensity');
title('Euclidean Distance R-Intensity Curve');
legend('Pixel values', 'Mean intensity', 'Location', 'best');
grid on;

%% ==================== 5. Plot 2D heatmap ====================
figure('Name', '2D Heatmap', 'Position', [100 100 700 600]);
x_coords = info.col_origin + (0:info.cols-1) * info.col_delta;
y_coords = info.row_origin + (0:info.rows-1) * info.row_delta;
imagesc(x_coords, y_coords, grid);
axis xy equal tight;
colorbar;
colormap(viridis());
title('2D Heatmap');
xlabel('X');
ylabel('Y');

%% ==================== Functions ====================

function [grid, info] = read_dat(path)
%READ_DAT  Parse optical .dat file (same format as the Python dat_parser).
%
%  File format:
%    GRID <rows> <cols>
%    <hole_value>
%    <row_delta> <col_delta>
%    <row_origin> <col_origin>
%    <data_matrix ...>
%
%  Lines starting with '!' are treated as comments and skipped.

    fid = fopen(path, 'r');
    if fid == -1
        error('Cannot open file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    % Read all non-empty, non-comment lines
    raw_lines = {};
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if isempty(line) || line(1) == '!'
            continue;
        end
        raw_lines{end+1} = line; %#ok<AGROW>
    end

    if numel(raw_lines) < 5
        error('DAT file incomplete: need at least header + data matrix.');
    end

    % Line 1: GRID rows cols
    tokens = regexp(raw_lines{1}, '\s+', 'split');
    if ~strcmpi(tokens{1}, 'GRID') || numel(tokens) < 3
        error('DAT file format error: first line must be "GRID rows cols".');
    end
    nrows = str2double(tokens{2});
    ncols = str2double(tokens{3});

    % Line 2: hole_value
    hole_value = str2double(regexp(raw_lines{2}, '\s+', 'split'));

    % Line 3: row_delta col_delta
    delta_tok = regexp(raw_lines{3}, '\s+', 'split');
    row_delta = str2double(delta_tok{1});
    col_delta = str2double(delta_tok{end});

    % Line 4: row_origin col_origin
    origin_tok = regexp(raw_lines{4}, '\s+', 'split');
    row_origin = str2double(origin_tok{1});
    col_origin = str2double(origin_tok{end});

    % Lines 5+: data matrix
    data_lines = raw_lines(5:end);
    data_str = strjoin(data_lines, '\n');
    grid = sscanf(data_str, '%f', [ncols, nrows])';  % transpose: row-major text -> MATLAB rows

    if ~isequal(size(grid), [nrows, ncols])
        error('Matrix dimension mismatch: header says %dx%d, got %dx%d.', ...
              nrows, ncols, size(grid,1), size(grid,2));
    end

    % Replace hole_value with NaN
    grid(grid == hole_value) = NaN;

    info.rows       = nrows;
    info.cols       = ncols;
    info.hole_value = hole_value;
    info.row_delta  = row_delta;
    info.col_delta  = col_delta;
    info.row_origin = row_origin;
    info.col_origin = col_origin;
end


function result = radius_mean(matrix, center_row, center_col, radius, mode, offset_row, offset_col)
%RADIUS_MEAN  Compute mean/std/count/min/max within a radius from center.
%  Coordinates are 1-based (MATLAB convention).
%  offset_row/col are subtracted before computing distance (matches Python).

    [nrows, ncols] = size(matrix);
    [rr, cc] = ndgrid(1:nrows, 1:ncols);
    dr = rr - offset_row - center_row;
    dc = cc - offset_col - center_col;

    if strcmpi(mode, 'manhattan')
        dist = abs(dr) + abs(dc);
    else
        dist = sqrt(dr.^2 + dc.^2);
    end

    mask   = (dist <= radius) & ~isnan(matrix);
    vals   = matrix(mask);

    if isempty(vals)
        result.mean  = NaN;
        result.std   = NaN;
        result.count = 0;
        result.min   = NaN;
        result.max   = NaN;
    else
        result.mean  = mean(vals);
        result.std   = std(vals);
        result.count = numel(vals);
        result.min   = min(vals);
        result.max   = max(vals);
    end
end


function [radii, means, counts] = radius_intensity_curve( ...
    matrix, center_row, center_col, max_radius, mode, offset_row, offset_col)
%RADIUS_INTENSITY_CURVE  Compute mean intensity at each radial bin.
%
%  Manhattan mode  — bins are integer distances 0, 1, 2, ..., max_radius.
%  Euclidean mode  — bins are uniformly spaced with step 0.01
%                    (i.e. n_points = max_radius / 0.01).
%
%  Matches the Python radius_intensity_curve() logic exactly.

    [nrows, ncols] = size(matrix);
    [rr, cc] = ndgrid(1:nrows, 1:ncols);
    dr = rr - offset_row - center_row;
    dc = cc - offset_col - center_col;

    if strcmpi(mode, 'manhattan')
        dist = abs(dr) + abs(dc);
    else
        dist = sqrt(dr.^2 + dc.^2);
    end

    valid = ~isnan(matrix);
    r = dist(valid);
    v = matrix(valid);

    if isempty(r)
        radii  = [];
        means  = [];
        counts = [];
        return;
    end

    if strcmpi(mode, 'manhattan')
        % Integer bins: 0, 1, 2, ..., max_radius
        max_r  = round(max_radius);
        radii  = (0:max_r)';
        means  = NaN(max_r+1, 1);
        counts = zeros(max_r+1, 1);
        r_int  = round(r);

        for i = 0:max_r
            idx = (r_int == i);
            cnt = nnz(idx);
            counts(i+1) = cnt;
            if cnt > 0
                means(i+1) = mean(v(idx));
            end
        end
    else
        % Uniform bins with step 0.01
        n_points = round(max_radius / 0.01);
        edges    = linspace(0, max_radius, n_points + 1);
        radii    = (edges(1:end-1) + edges(2:end)) / 2;
        means    = NaN(n_points, 1);
        counts   = zeros(n_points, 1);

        for i = 1:n_points
            if i < n_points
                idx = (r >= edges(i)) & (r < edges(i+1));
            else
                idx = (r >= edges(i)) & (r <= edges(i+1));
            end
            cnt = nnz(idx);
            counts(i) = cnt;
            if cnt > 0
                means(i) = mean(v(idx));
            end
        end
    end
end


function c = viridis()
%VIRIDIS  Approximate viridis colormap (256 entries).
%  Replicates the Plotly Viridis colorscale used in the Python app.

    M = [ ...
        0.267004  0.004874  0.329415; 0.269841  0.007856  0.339620;
        0.272655  0.011018  0.349858; 0.275426  0.014361  0.360129;
        0.278140  0.017886  0.370432; 0.280786  0.021590  0.380768;
        0.283354  0.025472  0.391136; 0.285834  0.029530  0.401535;
        0.288217  0.033763  0.411965; 0.290495  0.038170  0.422426;
        0.292662  0.042748  0.432917; 0.294710  0.047495  0.443438;
        0.296633  0.052409  0.453989; 0.298425  0.057488  0.464568;
        0.300082  0.062728  0.475175; 0.301599  0.068126  0.485809;
        0.302974  0.073679  0.496468; 0.304203  0.079384  0.507150;
        0.305285  0.085237  0.517855; 0.306218  0.091235  0.528580;
        0.307001  0.097374  0.539324; 0.307634  0.103651  0.550085;
        0.308116  0.110061  0.560862; 0.308447  0.116601  0.571653;
        0.308628  0.123266  0.582457; 0.308658  0.130053  0.593271;
        0.308538  0.136957  0.604094; 0.308269  0.143974  0.614924;
        0.307851  0.151100  0.625760; 0.307286  0.158330  0.636600;
        0.306576  0.165660  0.647442; 0.305722  0.173086  0.658283;
        0.304728  0.180602  0.669123; 0.303597  0.188204  0.679958;
        0.302332  0.195889  0.690788; 0.300935  0.203652  0.701611;
        0.299410  0.211489  0.712424; 0.297760  0.219397  0.723226;
        0.295988  0.227371  0.734015; 0.294099  0.235408  0.744790;
        0.292097  0.243503  0.755548; 0.289986  0.251653  0.766288;
        0.287769  0.259854  0.777007; 0.285451  0.268101  0.787703;
        0.283035  0.276391  0.798375; 0.280526  0.284720  0.809021;
        0.277927  0.293084  0.819639; 0.275242  0.301480  0.830227;
        0.272474  0.309904  0.840784; 0.269628  0.318353  0.851307;
        0.266707  0.326824  0.861794; 0.263714  0.335313  0.872244;
        0.260653  0.343817  0.882655; 0.257527  0.352332  0.893025;
        0.254340  0.360855  0.903354; 0.251096  0.369382  0.913639;
        0.247797  0.377912  0.923878; 0.244448  0.386441  0.934070;
        0.241051  0.394967  0.944214; 0.237610  0.403487  0.954307;
        0.234129  0.411998  0.964349; 0.230611  0.420498  0.974338;
        0.227059  0.428984  0.984272; 0.223477  0.437454  0.994150;
        0.219869  0.445906  0.996590; 0.216237  0.454337  0.991658;
        0.212585  0.462746  0.986727; 0.208916  0.471130  0.981798;
        0.205232  0.479490  0.976871; 0.201535  0.487822  0.971948;
        0.197826  0.496128  0.967028; 0.194108  0.504405  0.962113;
        0.190381  0.512654  0.957202; 0.186647  0.520873  0.952297;
        0.182908  0.529063  0.947397; 0.179164  0.537222  0.942504;
        0.175417  0.545350  0.937618; 0.171668  0.553447  0.932739;
        0.167918  0.561512  0.927869; 0.164168  0.569544  0.923008;
        0.160419  0.577544  0.918157; 0.156671  0.585511  0.913316;
        0.152927  0.593445  0.908486; 0.149186  0.601345  0.903668;
        0.145449  0.609212  0.898862; 0.141718  0.617045  0.894069;
        0.137993  0.624843  0.889290; 0.134275  0.632607  0.884526;
        0.130563  0.640336  0.879777; 0.126860  0.648030  0.875044;
        0.123165  0.655688  0.870328; 0.119479  0.663311  0.865630;
        0.115802  0.670899  0.860950; 0.112135  0.678451  0.856289;
        0.108478  0.685968  0.851649; 0.104831  0.693450  0.847028;
        0.101195  0.700897  0.842429; 0.097570  0.708308  0.837851;
        0.093956  0.715684  0.833296; 0.090354  0.723024  0.828765;
        0.086764  0.730329  0.824259; 0.083185  0.737599  0.819778;
        0.079618  0.744833  0.815323; 0.076064  0.752032  0.810894;
        0.072522  0.759196  0.806492; 0.068993  0.766324  0.802118;
        0.065476  0.773418  0.797772; 0.061972  0.780477  0.793455;
        0.058481  0.787501  0.789168; 0.055003  0.794491  0.784911;
        0.051538  0.801446  0.780685; 0.048086  0.808367  0.776491;
        0.044647  0.815254  0.772329; 0.041221  0.822107  0.768200;
        0.037809  0.828926  0.764104; 0.034410  0.835712  0.760043;
        0.031024  0.842464  0.756017; 0.027652  0.849183  0.752026;
        0.024292  0.855869  0.748072; 0.020946  0.862522  0.744155;
        0.017612  0.869142  0.740276; 0.014291  0.875730  0.736435;
        0.010983  0.882285  0.732634; 0.007688  0.888808  0.728873;
        0.004405  0.895299  0.725152; 0.001134  0.901758  0.721473];
    c = M;
end

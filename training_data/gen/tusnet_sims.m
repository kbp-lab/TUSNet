clear all;
close all;

addpath '/Users/X/MATLAB/k-Wave'
device = 'gpuArray-single';

%% Make Simulation Directories

mkdir tusnet_sims
mkdir tusnet_sims/inputs
mkdir tusnet_sims/pressure
mkdir tusnet_sims/mat

%% Plot Dimensions and Resolution

Lx = 6;                     % Simulation Height [cm]
Ly = 6;                     % Simulation Width [cm]
gs = 0.1171875;             % [mm] per grid point

Nx = Lx / (gs / 10);
Ny = Ly / (gs / 10);

%% Transducer and Input Signal

td_type = 0;                % Flat (0) or Curved (1)
n_e = 80;                   % Number of Elements
td_dia = 9;                 % Curved Transducer Diameter [cm]
    
source = 1e6;               % Peak Pressure [MPa]
freq = 0.5e6;               % Sonication Frequency [MHz]
cycles = 5;                 % Pulse Cycles

td_loc = [0.3 Lx/2];        % Transducer Location [cm]

%% Get Skulls and Targets

% Load predefined skulls ('slices'), TUSNet-predicted phase vectors ('phases'), and random targets ('targets')
load(path + "data/randomized_targets_500.mat");

% Alternatively, you can generate a grid of targets
% targets = targets(Nx, Ny);

% Normalization Factors
skull_norm = 2734.531494140625;
field_norms_norm = 725208.1250;
phase_norm = 1.8769439570576196e-05;

%% Run Simulations

% Simulation Options
sim_ground_truth = false;       % Simulate Ground Truth
show = false;                   % Show Plot
n_sims = size(phases, 1);       % Number of Sims to Run

gt_sims = zeros(n_sims, 512, 512);
tusnet_sims = zeros(n_sims, 512, 512);
dts = zeros(n_sims, 1);

computation_time = [];

% Simulation Loop
for m = 1 : n_sims

    % Extract Correct Slice
    slice = squeeze(slices(ceil(m / 50), :, :));   % 50 random targets per slice
    
    % Get Ground Truth and TUSNet Phases
    net_phase = squeeze(phases(m, :));

    % Convert Phases from Normalized Values to Time Delays
    net_phase = net_phase * phase_norm;

    % Target
    n = 0; % target counter, zero in this case since file names depend on 'm'
    target = convert_targets_2d(targets(m, 1), targets(m, 2));

    t_start = tic;

    % Run Simulation
    [p_max, p_max_tusnet, dt] = sim(slice, td_type, td_dia, td_loc, n_e, target, ...
                                    source, freq, cycles, Lx, Ly, gs, show, sim_ground_truth, ...
                                    m, n, net_phase, device);

    % Save Results
    save("tusnet_sims/mat/s" + string(m) + "p" + string(n) + ".mat", ...
        'p_max', 'p_max_tusnet', 'dt')

    gt_sims(m, :, :) = p_max;
    tusnet_sims(m, :, :) = p_max_tusnet;
    dts(m) = dt;

    t_end = toc(t_start);
    disp('Finished simulation in ' + string(t_end) + ' seconds')
    computation_time = [computation_time, t_end];

    if length(computation_time) >= 10
        disp('Average time over ' + string(length(computation_time)) + 'runs: ' + string(mean(computation_time)))
        break;
    end
    
end    

% Save all simulations
save("tusnet_sims/sims.mat", 'gt_sims', 'tusnet_sims', 'dts', '-v7.3')

%% Functions

% Simulate Pressure Field and Phase Aberration Corrections
function [p_max, p_max_tusnet, dt] = sim(skull, ...
    td_type, td_dia, td_loc, n_elements, target, source_strength, ...
    tone_burst_freq, tone_burst_cycles, Lx, Ly, gs, show, compute_init, ...
    m, n, tusnet_phase, device)

    % Input Parameters:
    % - skull: CT scan data of the skull
    % - td_type: Transducer type (0 for flat transducer, 1 for curved transducer)
    % - td_dia: Transducer diameter (applicable for curved transducer)
    % - td_loc: Transducer location (x,y coordinates)
    % - n_elements: Number of elements in the transducer
    % - target: Target location (applicable for phase aberration computation)
    % - source_strength: Strength of the ultrasound pulse
    % - tone_burst_freq: Frequency of the ultrasound pulse
    % - tone_burst_cycles: Number of cycles in the ultrasound pulse
    % - Lx: Length of the domain in the x-direction
    % - Ly: Length of the domain in the y-direction
    % - gs: Grid spacing
    % - show: Flag to indicate if plots should be displayed or saved as images (true or false)
    % - compute_init: Flag to indicate whether to compute ground truth pressure field (true or false)
    % - m: Index for current skull (used for saving figures)
    % - n: Index for current target (used for saving figures)
    % - gt_phase: Ground-truth phase correction
    % - tusnet_phase: TUSNet phase correction
    % - device: Data type for GPU computation ('gpuArray-single' for GPU, 'single' for CPU)

    % k-Wave Grid
    Nx = Lx / (gs / 10);
    Ny = Ly / (gs / 10);
    kgrid = kWaveGrid(Nx, gs*1e-3, Ny, gs*1e-3);
    
    %% Medium
    
    % Clip Skull to 25 HU
    skull_clipped = skull;
    skull_clipped(skull < 25) = 0;
    
    % Acoustic Parameters uing k-Wave Convention
    density_map = hounsfield2density(skull_clipped);
    density_map(density_map < 997) = 997;
    velocity_map = 1.33 * density_map + 167;

    % Constant Attenuation of 13.3 db/cm/MHz
    attenuation_map = 0.53 * ones(512, 512);
    attenuation_map(skull_clipped > 100) = 13.3;

    % Set Medium Parameters
    medium.sound_speed = velocity_map;
    medium.density = density_map;
    medium.alpha_coeff = attenuation_map;
    medium.alpha_power = 1.5;

    %% Show Acoustic Parameters

    figure(visible="off")
    subplot(2, 2, 1)
    imagesc(skull)
    axis image
    axis off
    colormap gray
    colorbar
    title("Original CT Scan (HU)")
    
    subplot(2, 2, 2)
    imagesc(medium.sound_speed)
    axis image
    axis off
    colormap gray
    colorbar
    title("Velocity Map (m/s)")
    
    subplot(2, 2, 3)
    imagesc(medium.density)
    axis image
    axis off
    colormap gray
    colorbar
    title("Density Map (kg/m³)")
    
    subplot(2, 2, 4)
    imagesc(medium.alpha_coeff)
    axis image
    axis off
    colormap gray
    colorbar
    title("Attenuation Map (dB/cm)")
    saveas(gcf, "tusnet_sims/inputs/skull_" + string(m) + ".png", 'png')
        
    %% Simulation Grid
    
    kgrid.makeTime(medium.sound_speed, [], .12 / min(medium.sound_speed(:)));
    dt = kgrid.dt;

    %% Transducer
    
    element_width = floor(Nx / (n_elements));

    vector_mask = zeros(Nx, Ny);
    
    if td_type == 0          % Flat Transducer
    
        td_x0 = round(td_loc(1) / (gs / 10));
        td_y0 = round((td_loc(2) / (gs / 10)) - (n_elements * element_width / 2));

        vector_mask(td_x0, td_y0 : element_width : td_y0 + n_elements * element_width - 1) = 1;
    
    elseif td_type == 1      % Curved Transducer

        td_loc = round(td_loc / (gs / 10));
        td_dia = round(td_dia / (gs / 10)) + 1;

        circle = makeArc([Nx Ny], td_loc, 8 * td_dia, td_dia, ...
            [Nx/2 Ny/2]);

        source_positions = circle == 1;
        vector_mask(source_positions) = 1;
    
    end
    
    input_signal = toneBurst(1/kgrid.dt, tone_burst_freq, tone_burst_cycles);

    %% TUSNet Phases

    % Convert to MATLAB Convention (Simulation Time Points vs. Milliseconds)
    tusnet_phase = round(tusnet_phase / dt);
    tusnet_phase(tusnet_phase < 0) = 0;

    %% Phase Aberration Correction

    % Test Source
    source.p_mask = zeros(Nx, Ny);
    source.p_mask(target) = 1;

    sensor.mask = vector_mask;
    sensor.record = {'p'};

    pac_strength = 10;  % [Pa]
    pac_cycles = 1;

    % Test Signal
    source.p = pac_strength * toneBurst(1/kgrid.dt, tone_burst_freq, pac_cycles);

    % Simulaton Arguments
    input_args = {'DisplayMask', source.p_mask, 'PMLInside', false, 'PlotPML', false, ...
        'DataCast', device, 'PMLSize', 14, 'PMLAlpha', 5, 'PlotSim', false};

    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

    % Compute Phase Delays
    [~, argmax] = max(sensor_data.p, [], 2);
    phase_shifts = max(argmax) - argmax;

    % Plot Phase Delays
    if show == true
        figure;
        bar(phase_shifts, 'BarWidth', 2);
        title('Corrected Phase Delays')
    else
        gt_phase = squeeze(gather(phase_shifts(:, 1)));
    end

    %% Initial Pressure Wave (Ground Truth)
    
    p_max = zeros(Nx, Ny);

    if compute_init == true

        % Monitor Pressure Wave from Transducer
        source.p_mask = vector_mask;
        sensor.mask = ones(Nx,Ny);
        sensor.record = {'p', 'p_max'};

        % Define Input with Phase Delays
        source.p = source_strength * toneBurst(1/kgrid.dt, tone_burst_freq, ...
                tone_burst_cycles, 'SignalOffset', gt_phase);
        
        % Simulation Arguments
        input_args = {'DisplayMask', source.p_mask, 'PMLInside', false, 'PlotPML', false, ... 
                    'DataCast', device, 'PMLSize', 14, 'PMLAlpha', 5, 'PlotSim', false};
        
        sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
        
        % Plot Initial Peak Pressure Wave Distribution
        p_max = reshape(sensor_data.p_max, Nx, Ny);
        cTitle = 'Ground Truth Peak Pressure';
        customPlot(cTitle, p_max / 1e6, Lx, Ly, gs, show, m, n);

    end

    % Optional Debugging Breakpoint to Check Ground Truth Pressure Field
    % keyboard
    
    %% TUSNet Corrected Pressure Wave
    
    % Monitor Pressure Wave from Transducer
    source.p_mask = vector_mask;
    sensor.mask = ones(Nx,Ny);
    sensor.record = {'p_max'};
    
    % Define Input with Phase Delays
    source.p = source_strength * toneBurst(1/kgrid.dt, tone_burst_freq, ...
                tone_burst_cycles, 'SignalOffset', tusnet_phase);
    
    % Simulation Arguments
    input_args = {'DisplayMask', source.p_mask, 'PMLInside', false, 'PlotPML', false, ... 
                'DataCast', device, 'PMLSize', 14, 'PMLAlpha', 5, 'PlotSim', false};

    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    
    % Plot Corrected Peak Pressure Distribution
    p_max_tusnet = gather(reshape(sensor_data.p_max, Nx, Ny));
    cTitle = 'TUSNet Peak Pressure';
    customPlot(cTitle, p_max_tusnet / 1e6, Lx, Ly, gs, show, m, n);

    % Optional Breakpoint to Check TUSNet Pressure Field
    % keyboard

end

% Custom Plot
function [] = customPlot(mTitle, data, Lx, Ly, gs, show, m, n)

    if show == true

        figure("Color", 'white');
        imagesc(data);
        axis square
        colormap parula

        c = colorbar;
        c.Label.String = "Pressure [MPa]";

        title(mTitle);
        xlabel('y [cm]');
        ylabel('x [cm]');

        Nx = Lx / (gs / 10);
        Ny = Lx / (gs / 10);
    
        set(gca,'XTick', 0:(10/gs):Nx);
        set(gca,'YTick', 0:(10/gs):Ny);
        set(gca, 'XTickLabel', 0:Lx);
        set(gca, 'YTickLabel', 0:Ly);
        set(gca, 'FontSize', 14)

    else

        Irgb = ind2rgb(im2uint8(data / max(data, [], 'all')), parula);
        imwrite(Irgb, "tusnet_sims/pressure/skull_" + string(m) + ...
            "_pos_" + string(n) + ".png")

    end

end

function [points] = targets(Nx, Ny)

    points = zeros(Nx, Ny);
    segment = Nx / 16;
    points(segment*(7:14), segment*(5:11)) = 1;
    points = find(points == 1);

end

function [target] = convert_targets_2d(target_x, target_y)

    points = zeros(512, 512);
    points(target_y, target_x) = 1;
    target = find(points == 1);

end

% Custom Plot for Phases
function [] = phasePlot(phases, mtitle, units)

    % Input:
    % - phases: Matrix of vectors or a single vector.
    %     - Matrix of vectors: Each row represents a vector of phases.
    %     - Single vector: Represents a single vector of phases.
    % - mtitle: Title(s) of the plot(s); string or array of strings
    % - units: Label for the y-axis units.

    % Check if input is a matrix or a vector
    if ismatrix(phases)
        numVectors = size(phases, 1);
    elseif isvector(phases)
        numVectors = 1;
        phases = phases(:);
    end

    % Calculate figure size
    screenSize = get(groot, 'ScreenSize');
    figureWidth = screenSize(3) * 0.6;  % 60% of screen width
    figureHeight = screenSize(4) * 0.35;  % 35% of screen height
    figureX = screenSize(3) * 0.2;  % 20% of screen width
    figureY = screenSize(4) * 0.2;  % 20% of screen height
    figurePos = [figureX, figureY, figureWidth, figureHeight];
        
    % Set up subplots
    figure('Color', 'white', 'Position', figurePos);
    
    for i = 1:numVectors
        subplot(1, numVectors, i);
        set(gca, 'FontSize', 14);

        % Plot the vector
        h = bar(phases(i, :));
        xlabel('Element');
        ylabel(units);
        
        if numVectors == 1
            title(mtitle);
        else
            title(mtitle(i));
        end
        
        box off;
        barColor = [0.4 0.6 0.8];
        set(h, 'FaceColor', barColor);
    end
    
end

function scope_grid_netcdf_inmemory_refactored(scope_main_dir, in_nc, out_nc)
% Refactored SCOPE grid runner with clear separation of configuration and computation
% Main orchestration function that coordinates configuration and model execution

    % if nargin < 3 || isempty(in_nc),  in_nc  = fullfile(experiment_dir,'input','grid_timeseries.nc'); end
    % if nargin < 4 || isempty(out_nc), out_nc = fullfile(experiment_dir,'output','grid_results.nc'); end
    
    % 1. Setup and Configuration
    config = setup_scope_environment(scope_main_dir, in_nc);
    
    % 2. Load input data
    input_data = load_input_data(in_nc);

    % 3. Process grid
    results = process_scope_grid(input_data, config);
    
    % % 3.5. Convert list-based accumulation to final arrays
    % results = convert_lists_to_arrays(results);
    
    % 4. Write output
    write_results_to_netcdf(results, out_nc, input_data.metadata);
end

%% ========================================================================
%% CONFIGURATION FUNCTIONS
%% ========================================================================

function config = setup_scope_environment(scope_main_dir, in_nc)
% Configure SCOPE environment, paths, constants, and options

    % Add SCOPE paths
    addpath(fullfile(scope_main_dir,'src','RTMs'));
    addpath(fullfile(scope_main_dir,'src','supporting'));
    addpath(fullfile(scope_main_dir,'src','fluxes'));
    addpath(fullfile(scope_main_dir,'src','IO'));
    
    % Load SCOPE constants and spectral bands
    config.constants = define_constants;
    config.spectral = define_bands;
    
    % Debug: Check spectral bands immediately after loading
    fprintf('DEBUG: wlF range after define_bands: %.1f-%.1f, length: %d\n', ...
        min(config.spectral.wlF), max(config.spectral.wlF), length(config.spectral.wlF));
    
    fprintf('wlF shape: %s\n', mat2str(size(config.spectral.wlF)));

    % Read NetCDF metadata and options
    I = ncinfo(in_nc);
    [config.optipar_file, config.soil_file, config.atmos_file, config.lidf_file, config.options] = read_global_opts(I);
    % print out options
    fprintf('SCOPE Options:\n');
    disp(config.options);


    config.options.simulation = 1;    % enforce time-series behavior
    config.options.saveCSV = 0;
    % Temporarily disable fluorescence to isolate the radiative transfer issues
    % config.options.calc_fluor = 0;
    
    % Load static resources
    config = load_static_resources(config);
    
    % Set fixed model components
    config.soilemp.SMC = 25;
    config.soilemp.film = 0.015;
    config.TDP = define_temp_response_biochem;
    config.integr = iff(config.options.lite==0,'angles_and_layers','layers');
end

function config = load_static_resources(config)
% Load static resources: optipar, soil spectra, atmosphere, LIDF
    
    % base_input = fullfile(experiment_dir,'input');
    
    % Load optical parameters
    if isempty(config.optipar_file), config.optipar_file = 'optipar.mat'; end
    % optipar_path = fullfile(base_input,'fluspect_parameters',config.optipar_file);
    optipar_path = config.optipar_file;
    if ~exist(optipar_path,'file'), error('optipar not found: %s', optipar_path); end
    load(optipar_path,'-mat'); % -> optipar
    config.optipar = optipar;
    
    % Load soil spectra (if needed)
    config.rsfile = [];
    if config.options.soilspectrum==0
        if isempty(config.soil_file), config.soil_file = 'soil_spectrum.dat'; end
        % soil_path = fullfile(base_input,'soil_spectra',config.soil_file);
        soil_path = config.soil_file;
        if ~exist(soil_path,'file'), error('soil spectrum not found: %s', soil_path); end
        config.rsfile = load(soil_path);
    end
    
    % Load atmosphere
    if isempty(config.atmos_file), config.atmos_file = 'atmos.dat'; end
    % atmos_path = fullfile(base_input,'radiationdata',config.atmos_file);
    atmos_path = config.atmos_file;
    if exist(atmos_path,'file')
        config.atmo = load_atmo(atmos_path, config.spectral.SCOPEspec);
    else
        warning('scope_grid:atmoMissing','Atmosphere file missing: %s; using placeholder', atmos_path);
        config.atmo = load_atmo_placeholder(config.spectral);
    end
    
    % Load LIDF (leaf inclination distribution function)
    config.lidf_mat = [];
    if ~isempty(config.lidf_file)
        % lf = fullfile(base_input,'leafangles',config.lidf_file);
        lf = config.lidf_file;
        if exist(lf,'file')
            try
                config.lidf_mat = dlmread(lf,'',3,0);
            catch ME
                warning(ME.identifier,'%s',ME.message);
                config.lidf_mat = [];
            end
        else
            warning('scope_grid:lidfMissing','LIDF file not found: %s', lf);
        end
    end
end

%% ========================================================================
%% INPUT DATA LOADING
%% ========================================================================

function input_data = load_input_data(in_nc)
% Load and organize input data from NetCDF file
    
    I = ncinfo(in_nc);
    [ny,nx,nt] = get_dims(I);
    fprintf('Input data dimensions: ny: %d nx: %d nt: %d\n', ny, nx, nt);
    input_data.metadata.ny = ny;
    input_data.metadata.nx = nx;
    input_data.metadata.nt = nt;
    input_data.metadata.ncinfo = I;
    
    % Load all variables with SCOPE defaults
    input_data.grid = load_grid_variables(in_nc, I, ny, nx, nt);
end

function G = load_grid_variables(in_nc, I, ny, nx, nt)
% Load all grid variables with proper SCOPE defaults
    
    G = struct();

    % Meteorological variables
    G.Rin = read_nc_var(in_nc,I,{'Rin'},ny,nx,nt,NaN);
    G.Rli = read_nc_var(in_nc,I,{'Rli'},ny,nx,nt,NaN);
    G.p   = read_nc_var(in_nc,I,{'p'},  ny,nx,nt,NaN);
    G.Ta  = read_nc_var(in_nc,I,{'Ta'}, ny,nx,nt,NaN);
    G.u   = read_nc_var(in_nc,I,{'u'},  ny,nx,nt,NaN);
    G.ea  = read_nc_var(in_nc,I,{'ea'}, ny,nx,nt,NaN);
    % G.RH  = read_nc_var(in_nc,I,{'RH'}, ny,nx,nt,NaN);   % Default RH = 50%
    % G.VPD = read_nc_var(in_nc,I,{'VPD'},ny,nx,nt,NaN); % Default VPD = 1.0 kPa
    
    % Geometry
    G.tts = read_nc_var(in_nc,I,{'tts','sza'},ny,nx,nt,30);
    G.tto = read_nc_var(in_nc,I,{'tto','vza'},ny,nx,nt,0);
    G.psi = read_nc_var(in_nc,I,{'psi','raa'},ny,nx,nt,0);
    
    % Vegetation
    G.LAI   = read_nc_var(in_nc,I,{'LAI','lai'},ny,nx,nt,3);
    G.Cab   = read_nc_var(in_nc,I,{'Cab','cab'},ny,nx,nt,40);
    G.Cca   = read_nc_var(in_nc,I,{'Cca','cca'},ny,nx,nt,10);
    G.Cdm   = read_nc_var(in_nc,I,{'Cdm','cm'}, ny,nx,nt,0.012);
    G.Cw    = read_nc_var(in_nc,I,{'Cw','cw'},  ny,nx,nt,0.009);
    G.Cs    = read_nc_var(in_nc,I,{'Cs','cbrown'},ny,nx,nt,0.0);
    G.Cant  = read_nc_var(in_nc,I,{'Cant'},ny,nx,nt,1.0);
    G.N     = read_nc_var(in_nc,I,{'N'},ny,nx,nt,1.5);
    G.hc    = read_nc_var(in_nc,I,{'hc'},  ny,nx,nt,2.0);
    G.LIDFa = read_nc_var(in_nc,I,{'LIDFa'},ny,nx,nt,-0.35);
    G.LIDFb = read_nc_var(in_nc,I,{'LIDFb'},ny,nx,nt,-0.15);
    
    % Soil
    G.SMC           = read_nc_var(in_nc,I,{'SMC'},ny,nx,nt,25);
    G.BSMBrightness = read_nc_var(in_nc,I,{'BSMBrightness'},ny,nx,nt,0.5);
    G.BSMlat        = read_nc_var(in_nc,I,{'BSMlat'},ny,nx,nt,25);
    G.BSMlon        = read_nc_var(in_nc,I,{'BSMlon'},ny,nx,nt,45);
    
    % Biochemistry
    G.Vcmax25        = read_nc_var(in_nc,I,{'Vcmax25'},ny,nx,nt,60);
    G.BallBerrySlope = read_nc_var(in_nc,I,{'BallBerrySlope'},ny,nx,nt,8);
    
    % Optional scalars
    G.z  = read_nc_var(in_nc,I,{'z'}, ny,nx,1,5);
    G.Ca = read_nc_var(in_nc,I,{'Ca'},ny,nx,1,410);
end

%% ========================================================================
%% MAIN PROCESSING FUNCTION
%% ========================================================================

function results = process_scope_grid(input_data, config)
% Main processing loop - separated from configuration
    
    G = input_data.grid;
    ny = input_data.metadata.ny;
    nx = input_data.metadata.nx;
    nt = input_data.metadata.nt;
    

    acc = initialize_accumulator(ny, nx, nt, config);

    % Main processing loops
    for iy = 1:ny
        for ix = 1:nx
            fprintf('Processing pixel (x=%d, y=%d)\n', ix, iy);
            % % Extract per-pixel scalars
            % z  = pick1(G.z, iy, ix, 1, 5);
            % Ca = pick1(G.Ca,iy, ix, 1, 410);

            % 'z': 10.0,     # Measurement height of meteorological data (m)
            % 'Ca': 380.0,   # Atmospheric CO2 concentration (ppm)
            % 'Oa': 209.0,   # Atmospheric O2 concentration (per mille)
            z = 5;
            Ca = 410;
            Oa = 209;
            for t = 1:nt
                % fprintf('Processing pixel (x=%d, y=%d, t=%d)\n', ix, iy, t);
                
                % % Debug: Check variable dimensions before accessing
                % if t == 1 && iy == 1 && ix == 1
                %     fprintf('Debug - Variable dimensions:\n');
                %     fprintf('  nt (loop limit): %d\n', nt);
                %     fprintf('  G.LAI size: [%s]\n', num2str(size(G.LAI)));
                %     fprintf('  G.Cab size: [%s]\n', num2str(size(G.Cab)));
                % end
                
                % % Check required variables
                % LAI = pick1(G.LAI,iy,ix,t,NaN);
                % Cab = pick1(G.Cab,iy,ix,t,NaN);
                % if isnan(LAI) || isnan(Cab), continue; end
                
                % Check required variables
                LAI = pick1(G.LAI,iy,ix,t,NaN);
                Rin = pick1(G.Rin,iy,ix,t,NaN);
                if isnan(LAI) || isnan(Rin), continue; end

                % Configure pixel-specific parameters
                pixel_params = configure_pixel_parameters(G, iy, ix, t, z, Ca, Oa, config);
                
                % Run SCOPE model
                pixel_results = run_scope_model(pixel_params, config);

                acc = accumulate_results(acc, pixel_results, iy, ix, t);
                
            end
        end
    end
    
    results = acc;
end

%% ========================================================================
%% PARAMETER CONFIGURATION
%% ========================================================================

function params = configure_pixel_parameters(G, iy, ix, t, z, Ca, Oa, config)
% Configure all parameters for a single pixel
    
    % Extract leaf area index and chlorophyll content
    LAI = pick1(G.LAI,iy,ix,t,NaN);
    Cab = pick1(G.Cab,iy,ix,t,NaN);
    
    % Configure meteorology
    params.meteo = configure_meteorology(G, iy, ix, t, z, Ca, Oa);
    
    % Configure geometry
    params.angles = configure_geometry(G, iy, ix, t);
    
    % Configure leaf biochemistry
    params.leafbio = configure_leaf_biochemistry(G, iy, ix, t, Cab, config);
    
    % Configure canopy structure
    params.canopy = configure_canopy_structure(G, iy, ix, t, LAI, z, config);
    
    % Configure soil properties
    params.soil = configure_soil_properties(G, iy, ix, t, config);
    
    % Configure mSCOPE parameters
    params.mly = configure_mscope_parameters(params.leafbio, params.canopy);
    
    % Time information
    params.k = t;
    params.xyt.t = t;
    params.xyt.x = ix;
    params.xyt.y = iy;
    params.xyt.year = 2000;
end

function meteo = configure_meteorology(G, iy, ix, t, z, Ca, Oa)
% Configure meteorological parameters with SCOPE defaults
    
    meteo.Rin = pick1(G.Rin,iy,ix,t,NaN);
    meteo.Rli = pick1(G.Rli,iy,ix,t,NaN);
    
    % Convert pressure from hPa to Pa (SCOPE expects Pa internally)
    p_hpa = pick1(G.p,  iy,ix,t,NaN);
    if ~isnan(p_hpa)
        meteo.p = p_hpa * 100;  % hPa to Pa
    else
        meteo.p = NaN;
    end
    
    meteo.Ta  = pick1(G.Ta, iy,ix,t,NaN);
    
    % Convert vapor pressure from hPa to Pa (SCOPE expects Pa internally)
    ea_hpa = pick1(G.ea, iy,ix,t,NaN);
    if ~isnan(ea_hpa)
        meteo.ea = ea_hpa * 100;  % hPa to Pa
    else
        meteo.ea = NaN;
    end
    
    meteo.u   = pick1(G.u,  iy,ix,t,NaN);

    % meteo.RH  = pick1(G.RH, iy,ix,t,NaN);
    % meteo.VPD = pick1(G.VPD,iy,ix,t,NaN);

    meteo.Ca  = Ca;     % SCOPE default: 410
    meteo.Oa  = Oa;     % SCOPE default: 209
    meteo.z   = z;      % SCOPE default: 5
end

function angles = configure_geometry(G, iy, ix, t)
% Configure sun-sensor geometry
    
    angles.tts = pick1(G.tts,iy,ix,t,30);  % SCOPE default: 30
    angles.tto = pick1(G.tto,iy,ix,t,0);   % SCOPE default: 0  
    angles.psi = pick1(G.psi,iy,ix,t,0);   % SCOPE default: 0
end

function leafbio = configure_leaf_biochemistry(G, iy, ix, t, Cab, config)
% Configure leaf biochemical parameters with SCOPE defaults
    
    leafbio.Cab = Cab;
    
    % Handle Cca default (0.25*Cab when not provided)
    Cca = pick1(G.Cca,iy,ix,t,NaN);
    if isnan(Cca), Cca = 0.25*Cab; end
    leafbio.Cca = Cca;
    
    % Structural parameters
    leafbio.Cdm  = pick1(G.Cdm,iy,ix,t,0.012);
    leafbio.Cw   = pick1(G.Cw, iy,ix,t,0.009);
    leafbio.Cs   = pick1(G.Cs, iy,ix,t,0.0);
    leafbio.Cant = pick1(G.Cant,iy,ix,t,1.0);
    leafbio.N    = pick1(G.N,   iy,ix,t,1.5);
    leafbio.Cbc  = 0.0;   % SCOPE default: 0
    leafbio.Cp   = 0.0;   % SCOPE default: 0
    
    % Thermal and fluorescence
    leafbio.rho_thermal = 0.01;  % SCOPE default: 0.01
    leafbio.tau_thermal = 0.01;  % SCOPE default: 0.01
    leafbio.fqe = 0.01;          % SCOPE default: 0.01
    
    % Photosynthesis
    leafbio.Vcmax25 = pick1(G.Vcmax25,iy,ix,t,60);
    leafbio.BallBerrySlope = pick1(G.BallBerrySlope,iy,ix,t,8);
    leafbio.RdPerVcmax25 = 0.015;   % SCOPE default: 0.015
    leafbio.BallBerry0 = 0.01;      % SCOPE default: 0.01
    leafbio.stressfactor = 1.0;     % SCOPE default: 1.0
    leafbio.Kn0 = 2.48;             % SCOPE default: 2.48
    leafbio.Knalpha = 2.83;         % SCOPE default: 2.83
    leafbio.Knbeta = 0.114;         % SCOPE default: 0.114
    
    % Photosynthetic pathway
    leafbio.Type = 0;  % Default to C3
    if leafbio.Type
        leafbio.Type = 'C4';
    else
        leafbio.Type = 'C3';
    end
    
    % Temperature response
    leafbio.TDP = config.TDP;
end

function canopy = configure_canopy_structure(G, iy, ix, t, LAI, z, config)
% Configure canopy structural parameters
    
    canopy.LAI = LAI;
    canopy.hc  = pick1(G.hc,iy,ix,t,2.0);      % SCOPE default: 2.0
    canopy.z   = z;
    canopy.leafwidth = 0.1;                    % SCOPE default: 0.1
    
    % Configure leaf inclination distribution
    if ~isempty(config.lidf_mat)
        canopy.lidf = config.lidf_mat;
    else
        canopy.LIDFa = -0.35;  % SCOPE default: -0.35
        canopy.LIDFb = -0.15;  % SCOPE default: -0.15
        canopy.lidf  = leafangles(canopy.LIDFa, canopy.LIDFb);
    end
    
    % Discretization parameters
    canopy.nlincl = 13; 
    canopy.nlazi = 36;
    canopy.litab  = [5:10:75 81:2:89]';
    canopy.lazitab= (5:10:355);
    canopy.nlayers= max(2, ceil(10*canopy.LAI) + ((config.options.MoninObukhov)*60));
    
    nl = canopy.nlayers;
    canopy.xl = [0; (-1/nl : -1/nl : -1)'];
    
    % Hot spot parameter
    canopy.hot = canopy.leafwidth / canopy.hc;
    
    % Aerodynamic parameters
    canopy.CR = 0.35;      % SCOPE default: 0.35
    canopy.CD1 = 20.6;     % SCOPE default: 20.6
    canopy.Psicor = 0.2;   % SCOPE default: 0.2
    canopy.kV = 0.64;      % SCOPE default: 0.64
    canopy.Cd = 0.3;       % SCOPE default: 0.3
    canopy.rwc = 0.0;      % SCOPE default: 0.0
end

function soil = configure_soil_properties(G, iy, ix, t, config)
% Configure soil properties with SCOPE defaults
    
    soil.SMC = pick1(G.SMC,iy,ix,t,25);         % SCOPE default: 25
    soil.rs_thermal = 0.06;                     % SCOPE default: 0.06
    soil.rss = 500;                             % SCOPE default: 500
    soil.rbs = 10;                              % SCOPE default: 10
    soil.CSSOIL = 0.01;                         % SCOPE default: 0.01
    
    % Convert SMC units if needed
    if mean(soil.SMC) > 1
        soil.SMC = soil.SMC / 100;  % SMC from [0 100] to [0 1]
    end
    
    % Configure thermal properties based on soil_heat_method
    if config.options.soil_heat_method == 1
        soil.GAM = Soil_Inertia1(soil.SMC);
    else
        soil.cs = 1180;        % SCOPE default: 1180
        soil.rhos = 1800;      % SCOPE default: 1800
        soil.lambdas = 1.55;   % SCOPE default: 1.55
        soil.GAM = Soil_Inertia0(soil.cs, soil.rhos, soil.lambdas);
    end
    
    % Calculate soil resistances if enabled
    if config.options.calc_rss_rbs
        LAI_val = pick1(G.LAI, iy, ix, t, 3);  % Get LAI for this pixel
        [soil.rss, soil.rbs] = calc_rssrbs(soil.SMC, LAI_val, soil.rbs);
    end
    
    % Configure soil spectrum
    if config.options.soilspectrum==0
        if ~isfield(soil,'spectrum') || isempty(soil.spectrum), soil.spectrum = 1; end
        soil.refl = config.rsfile(:, soil.spectrum+1);
    else
        soil.BSMBrightness = pick1(G.BSMBrightness,iy,ix,t,0.5);
        soil.BSMlat = pick1(G.BSMlat,iy,ix,t,25);
        soil.BSMlon = pick1(G.BSMlon,iy,ix,t,45);
        soil.refl = BSM(soil, config.optipar, config.soilemp);
    end
    soil.refl(config.spectral.IwlT) = soil.rs_thermal;
end

function mly = configure_mscope_parameters(leafbio, canopy)
% Configure mSCOPE multi-layer parameters (single layer for grid processing)
    
    mly.nly    = 1;
    mly.pLAI   = canopy.LAI; 
    mly.totLAI = canopy.LAI;
    mly.pCab   = leafbio.Cab; 
    mly.pCca   = leafbio.Cca; 
    mly.pCdm   = leafbio.Cdm;
    mly.pCw    = leafbio.Cw;  
    mly.pCs    = leafbio.Cs;  
    mly.pN     = leafbio.N;
end

%% ========================================================================
%% CORE MODEL CALCULATION
%% ========================================================================

function results = run_scope_model(params, config)
% Independent SCOPE model calculation function
% This function contains the core SCOPE computations separated from configuration
    
    % Calculate aerodynamic parameters
    [params.canopy.zo, params.canopy.d] = zo_and_d(params.soil, params.canopy, config.constants);
    
    % Calculate leaf optical properties
    results.leafopt = calculate_leaf_optics(params, config);
    
    % Run radiative transfer model
    [results.rad, results.gap] = run_radiative_transfer(params, config, results.leafopt);
    
    % Run energy balance model
    [results.iter, results.rad, results.thermal, results.soil, results.bcu, results.bch, ...
     results.fluxes, results.resistance, results.meteo] = run_energy_balance(params, config, results);
    
    if results.iter.counter > 99
        warning('Energy balance did not converge within 99 iterations (iter=%d)', results.iter.counter);

        % Print detailed input parameters for non-convergent case
        fprintf('\n=== NON-CONVERGENCE DIAGNOSTIC: ALL INPUT PARAMETERS ===\n');
        fprintf('Grid coordinates: y=%d, x=%d, t=%d\n', params.xyt.y, params.xyt.x, params.xyt.t);
        
        % Meteorological parameters
        fprintf('\n--- Meteorological Parameters ---\n');
        fprintf('Incoming shortwave radiation (Rin): %.2f W/m²\n', params.meteo.Rin);
        fprintf('Incoming longwave radiation (Rli): %.2f W/m²\n', params.meteo.Rli);
        fprintf('Air temperature (Ta): %.2f °C\n', params.meteo.Ta);
        fprintf('Atmospheric pressure (p): %.1f Pa\n', params.meteo.p);
        fprintf('Vapor pressure (ea): %.1f Pa\n', params.meteo.ea);
        fprintf('Wind speed (u): %.2f m/s\n', params.meteo.u);
        fprintf('CO2 concentration (Ca): %.1f ppm\n', params.meteo.Ca);
        fprintf('O2 concentration (Oa): %.1f mmol/mol\n', params.meteo.Oa);
        fprintf('Measurement height (z): %.1f m\n', params.meteo.z);
        
        % Canopy structure parameters - ALL FIELDS
        fprintf('\n--- Canopy Structure Parameters ---\n');
        fprintf('Leaf Area Index (LAI): %.3f\n', params.canopy.LAI);
        fprintf('Canopy height (hc): %.2f m\n', params.canopy.hc);
        fprintf('Measurement height (z): %.1f m\n', params.canopy.z);
        fprintf('Leaf width: %.3f m\n', params.canopy.leafwidth);
        fprintf('Leaf inclination distribution (LIDFa): %.3f\n', params.canopy.LIDFa);
        fprintf('Leaf inclination distribution (LIDFb): %.3f\n', params.canopy.LIDFb);
        fprintf('Number of inclination classes: %d\n', params.canopy.nlincl);
        fprintf('Number of azimuth classes: %d\n', params.canopy.nlazi);
        fprintf('Number of layers: %d\n', params.canopy.nlayers);
        fprintf('Hot spot parameter: %.4f\n', params.canopy.hot);
        fprintf('Roughness length (zo): %.4f m\n', params.canopy.zo);
        fprintf('Displacement height (d): %.4f m\n', params.canopy.d);
        fprintf('Crown ratio (CR): %.2f\n', params.canopy.CR);
        fprintf('Drag coefficient (CD1): %.1f\n', params.canopy.CD1);
        fprintf('Stability correction (Psicor): %.2f\n', params.canopy.Psicor);
        fprintf('Extinction coefficient (kV): %.2f\n', params.canopy.kV);
        fprintf('Drag coefficient (Cd): %.2f\n', params.canopy.Cd);
        fprintf('Relative water content (rwc): %.3f\n', params.canopy.rwc);
        
        % Leaf biochemistry parameters - ALL FIELDS
        fprintf('\n--- Leaf Biochemistry Parameters ---\n');
        fprintf('Chlorophyll a+b content (Cab): %.2f μg/cm²\n', params.leafbio.Cab);
        fprintf('Carotenoid content (Cca): %.2f μg/cm²\n', params.leafbio.Cca);
        fprintf('Brown pigment content (Cdm): %.4f g/cm²\n', params.leafbio.Cdm);
        fprintf('Equivalent water thickness (Cw): %.4f cm\n', params.leafbio.Cw);
        fprintf('Leaf mass per area (Cs): %.4f g/cm²\n', params.leafbio.Cs);
        fprintf('Anthocyanin content (Cant): %.3f μg/cm²\n', params.leafbio.Cant);
        fprintf('Leaf structure parameter (N): %.3f\n', params.leafbio.N);
        fprintf('Carotenoid breakdown products (Cbc): %.3f\n', params.leafbio.Cbc);
        fprintf('Protein content (Cp): %.3f\n', params.leafbio.Cp);
        fprintf('Thermal reflectance (rho_thermal): %.3f\n', params.leafbio.rho_thermal);
        fprintf('Thermal transmittance (tau_thermal): %.3f\n', params.leafbio.tau_thermal);
        fprintf('Fluorescence quantum efficiency (fqe): %.3f\n', params.leafbio.fqe);
        fprintf('Maximum carboxylation rate (Vcmax25): %.2f μmol/m²/s\n', params.leafbio.Vcmax25);
        fprintf('Ball-Berry slope: %.2f\n', params.leafbio.BallBerrySlope);
        fprintf('Respiration/Vcmax ratio: %.4f\n', params.leafbio.RdPerVcmax25);
        fprintf('Ball-Berry intercept: %.4f\n', params.leafbio.BallBerry0);
        fprintf('Stress factor: %.2f\n', params.leafbio.stressfactor);
        fprintf('Fluorescence parameters (Kn0/alpha/beta): %.2f / %.2f / %.3f\n', ...
                params.leafbio.Kn0, params.leafbio.Knalpha, params.leafbio.Knbeta);
        if isfield(params.leafbio, 'Type')
            if isnumeric(params.leafbio.Type)
                if params.leafbio.Type == 0
                    fprintf('Photosynthetic pathway: C3\n');
                else
                    fprintf('Photosynthetic pathway: C4\n');
                end
            else
                fprintf('Photosynthetic pathway: %s\n', params.leafbio.Type);
            end
        end
        
        % Soil parameters - ALL FIELDS
        fprintf('\n--- Soil Parameters ---\n');
        fprintf('Soil moisture content (SMC): %.3f m³/m³\n', params.soil.SMC);
        fprintf('Thermal reflectance (rs_thermal): %.3f\n', params.soil.rs_thermal);
        fprintf('Soil resistance (rss): %.1f s/m\n', params.soil.rss);
        fprintf('Boundary resistance (rbs): %.1f s/m\n', params.soil.rbs);
        fprintf('Soil carbon content (CSSOIL): %.4f\n', params.soil.CSSOIL);
        fprintf('Soil thermal inertia (GAM): %.2f\n', params.soil.GAM);
        if isfield(params.soil, 'cs')
            fprintf('Soil heat capacity (cs): %.0f J/kg/K\n', params.soil.cs);
            fprintf('Soil density (rhos): %.0f kg/m³\n', params.soil.rhos);
            fprintf('Soil thermal conductivity (lambdas): %.2f W/m/K\n', params.soil.lambdas);
        end
        if isfield(params.soil, 'BSMBrightness')
            fprintf('BSM Brightness: %.3f\n', params.soil.BSMBrightness);
            fprintf('BSM latitude: %.1f°\n', params.soil.BSMlat);
            fprintf('BSM longitude: %.1f°\n', params.soil.BSMlon);
        end
        
        % Angular parameters - ALL FIELDS
        fprintf('\n--- Solar/Viewing Geometry ---\n');
        fprintf('Solar zenith angle (tts): %.2f°\n', params.angles.tts);
        fprintf('Solar azimuth angle (psi): %.2f°\n', params.angles.psi);
        fprintf('Viewing zenith angle (tto): %.2f°\n', params.angles.tto);
        
        % mSCOPE parameters - ALL FIELDS
        fprintf('\n--- mSCOPE Multi-layer Parameters ---\n');
        fprintf('Number of layers (nly): %d\n', params.mly.nly);
        fprintf('Layer LAI (pLAI): %.3f\n', params.mly.pLAI);
        fprintf('Total LAI (totLAI): %.3f\n', params.mly.totLAI);
        fprintf('Layer Cab (pCab): %.2f μg/cm²\n', params.mly.pCab);
        fprintf('Layer Cca (pCca): %.2f μg/cm²\n', params.mly.pCca);
        fprintf('Layer Cdm (pCdm): %.4f g/cm²\n', params.mly.pCdm);
        fprintf('Layer Cw (pCw): %.4f cm\n', params.mly.pCw);
        fprintf('Layer Cs (pCs): %.4f g/cm²\n', params.mly.pCs);
        fprintf('Layer N (pN): %.3f\n', params.mly.pN);
        
        fprintf('=== END NON-CONVERGENCE DIAGNOSTIC ===\n\n');
        
    end
    % Run optional calculations
    results = run_optional_calculations(params, config, results);
    
    % Calculate derived products
    results = calculate_derived_products(params, config, results);
    
    % Store metadata
    results.xyt = params.xyt;
    results.k = params.k;
end

function leafopt = calculate_leaf_optics(params, config)
% Calculate leaf optical properties using fluspect
    
    nl = params.canopy.nlayers;
    
    % Set thermal emissivity
    params.leafbio.emis = 1 - params.leafbio.rho_thermal - params.leafbio.tau_thermal;
    params.leafbio.V2Z = 0;

    % Safeguard: fluspect_mSCOPE divides by sum(mly.pLAI); avoid zero / NaN when LAI==0
    if (~isfield(params,'mly')) || (~isfield(params.mly,'totLAI')) || ~isfinite(params.mly.totLAI) || params.mly.totLAI <= 0
        tiny_lai = 1e-10;          % very small placeholder LAI
        warning('calculate_leaf_optics:ZeroLAI', ...
            'Zero/invalid LAI (%.4g) replaced with tiny placeholder (%.1e) for optics only.', params.canopy.LAI, tiny_lai);
        params.mly.pLAI = tiny_lai;
        params.mly.totLAI = tiny_lai;
        if ~isfield(params.mly,'nly') || params.mly.nly < 1
            params.mly.nly = 1;
        end
        % Ensure required biochemical arrays exist for nly=1
        flds = {'pCab','pCca','pCdm','pCw','pCs','pN'};
        for fi = 1:numel(flds)
            if ~isfield(params.mly,flds{fi}) || isempty(params.mly.(flds{fi}))
                % Copy from leafbio
                baseName = flds{fi}(2:end); % remove leading 'p'
                if isfield(params.leafbio, baseName(1:3)) % not reliable; fallback direct mapping below
                end
            end
        end
        if ~isfield(params.mly,'pCab'), params.mly.pCab = params.leafbio.Cab; end
        if ~isfield(params.mly,'pCca'), params.mly.pCca = params.leafbio.Cca; end
        if ~isfield(params.mly,'pCdm'), params.mly.pCdm = params.leafbio.Cdm; end
        if ~isfield(params.mly,'pCw'),  params.mly.pCw  = params.leafbio.Cw;  end
        if ~isfield(params.mly,'pCs'),  params.mly.pCs  = params.leafbio.Cs;  end
        if ~isfield(params.mly,'pN'),   params.mly.pN   = params.leafbio.N;   end
    end
    
    try
        % Calculate leaf optical properties
        leafopt = fluspect_mSCOPE(params.mly, config.spectral, params.leafbio, config.optipar, nl);
        leafopt.refl(:, config.spectral.IwlT) = params.leafbio.rho_thermal;
        leafopt.tran(:, config.spectral.IwlT) = params.leafbio.tau_thermal;
    catch ME
        fprintf('\n=== ERROR during leaf optics computation ===\n');
        fprintf('Message: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('Top stack frame: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        % Spectral info
        if isfield(config,'spectral')
            sp = config.spectral;
            safe_rng = @(v) (isempty(v) || ~isnumeric(v)) * NaN + ( ~isempty(v) && isnumeric(v) ) * 0; %#ok<NASGU>
            fprintf('\n-- Spectral --\n');
            if isfield(sp,'wlS'), fprintf('wlS: n=%d min=%.2f max=%.2f\n', numel(sp.wlS), min(sp.wlS), max(sp.wlS)); end
            if isfield(sp,'wlF'), fprintf('wlF: n=%d min=%.2f max=%.2f\n', numel(sp.wlF), min(sp.wlF), max(sp.wlF)); end
            if isfield(sp,'IwlT'), fprintf('IwlT count: %d\n', numel(sp.IwlT)); end
            if isfield(sp,'IwlF'), fprintf('IwlF count: %d\n', numel(sp.IwlF)); end
            if isfield(sp,'IwlP'), fprintf('IwlP count: %d\n', numel(sp.IwlP)); end
        end
        % Options
        fprintf('\n-- Options --\n');
        disp(config.options);
        % Canopy
        fprintf('\n-- Canopy --\n');
        if isstruct(params.canopy)
            can_flds = fieldnames(params.canopy);
            for k=1:numel(can_flds)
                val = params.canopy.(can_flds{k});
                if isnumeric(val) && isscalar(val)
                    fprintf('  %s: %.6g\n', can_flds{k}, val);
                elseif isnumeric(val)
                    sz = size(val); fprintf('  %s: size=%s\n', can_flds{k}, mat2str(sz));
                else
                    fprintf('  %s: [%s]\n', can_flds{k}, class(val));
                end
            end
        end
        % Leaf bio
        fprintf('\n-- Leafbio --\n');
        if isstruct(params.leafbio)
            bio_flds = fieldnames(params.leafbio);
            for k=1:numel(bio_flds)
                val = params.leafbio.(bio_flds{k});
                if isnumeric(val) && isscalar(val)
                    fprintf('  %s: %.6g\n', bio_flds{k}, val);
                elseif isnumeric(val)
                    sz = size(val); fprintf('  %s: size=%s\n', bio_flds{k}, mat2str(sz));
                else
                    fprintf('  %s: [%s]\n', bio_flds{k}, class(val));
                end
            end
        end
        % mly
        fprintf('\n-- mly (multi-layer) --\n');
        if isstruct(params.mly)
            mly_flds = fieldnames(params.mly);
            for k=1:numel(mly_flds)
                val = params.mly.(mly_flds{k});
                if isnumeric(val) && isscalar(val)
                    fprintf('  %s: %.6g\n', mly_flds{k}, val);
                elseif isnumeric(val)
                    sz = size(val); fprintf('  %s: size=%s\n', mly_flds{k}, mat2str(sz));
                else
                    fprintf('  %s: [%s]\n', mly_flds{k}, class(val));
                end
            end
        end
        fprintf('  nl (canopy.nlayers): %d\n', nl);
        fprintf('=== END LEAF OPTICS ERROR DIAGNOSTIC ===\n\n');
        rethrow(ME);
    end
    
    % Calculate xanthophyll absorption if enabled
    if config.options.calc_xanthophyllabs
        params.leafbio.V2Z = 1;
        leafoptZ = fluspect_mSCOPE(params.mly, config.spectral, params.leafbio, config.optipar, nl);
        leafopt.reflZ = leafopt.refl; 
        leafopt.tranZ = leafopt.tran;
        leafopt.reflZ(:, config.spectral.IwlP) = leafoptZ.refl(:, config.spectral.IwlP);
        leafopt.tranZ(:, config.spectral.IwlP) = leafoptZ.tran(:, config.spectral.IwlP);
    end
end

function [rad, gap] = run_radiative_transfer(params, config, leafopt)
% Run SCOPE radiative transfer model (RTMo)
    
    % % Debug: Check for NaN/Inf in critical inputs
    % fprintf('RTMo Input Debugging:\n');
    % fprintf('  Meteo - Rin: %.2f, Ta: %.2f, p: %.2f, ea: %.2f, u: %.2f\n', ...
    %         params.meteo.Rin, params.meteo.Ta, params.meteo.p, params.meteo.ea, params.meteo.u);
    % fprintf('  Canopy - LAI: %.3f, hc: %.2f\n', params.canopy.LAI, params.canopy.hc);
    % fprintf('  Angles - tts: %.1f, tto: %.1f, psi: %.1f\n', ...
    %         params.angles.tts, params.angles.tto, params.angles.psi);
    % fprintf('  Leafbio - Cab: %.2f, Cw: %.4f, Cdm: %.4f, N: %.2f\n', ...
    %         params.leafbio.Cab, params.leafbio.Cw, params.leafbio.Cdm, params.leafbio.N);
    
    % % Check for NaN/Inf values
    % if any(isnan([params.meteo.Rin, params.meteo.Ta, params.meteo.p, params.meteo.ea, params.meteo.u]))
    %     warning('NaN detected in meteorological inputs!');
    % end
    % if any(isinf([params.meteo.Rin, params.meteo.Ta, params.meteo.p, params.meteo.ea, params.meteo.u]))
    %     warning('Inf detected in meteorological inputs!');
    % end
    % if any(isnan([params.canopy.LAI, params.canopy.hc]))
    %     warning('NaN detected in canopy inputs!');
    % end
    % if any(isnan([params.leafbio.Cab, params.leafbio.Cw, params.leafbio.Cdm, params.leafbio.N]))
    %     warning('NaN detected in leaf biochemistry inputs!');
    % end
    
    [rad, gap, ~] = RTMo(config.spectral, config.atmo, params.soil, leafopt, ...
                         params.canopy, params.angles, config.constants, params.meteo, config.options);
end

function [iter, rad, thermal, soil, bcu, bch, fluxes, resistance, meteo] = run_energy_balance(params, config, results)
% Run SCOPE energy balance model
    
    % Ensure leafbio has the emis field set
    params.leafbio.emis = 1 - params.leafbio.rho_thermal - params.leafbio.tau_thermal;
    
    [iter, rad, thermal, soil, bcu, bch, fluxes, resistance, meteo] = ...
        ebal(config.constants, config.options, results.rad, results.gap, params.meteo, ...
             params.soil, params.canopy, params.leafbio, params.k, params.xyt, config.integr);
end

function results = run_optional_calculations(params, config, results)
% Run optional SCOPE calculations (fluorescence, xanthophyll, thermal)
    
    % Fluorescence calculation
    if config.options.calc_fluor

       
        
        % % Debug: Print all RTMf input parameters
        % fprintf('=== RTMf Input Debug ===\n');
        
        % % Spectral information
        % fprintf('Spectral Data:\n');
        % fprintf('  Solar bands (wlS): %d bands, range %.1f-%.1f nm\n', ...
        %         length(config.spectral.wlS), min(config.spectral.wlS), max(config.spectral.wlS));
        % fprintf('  Fluor bands (wlF): %d bands, range %.1f-%.1f nm\n', ...
        %         length(config.spectral.wlF), min(config.spectral.wlF), max(config.spectral.wlF));
        % fprintf('  First 5 wlS: [%s]\n', num2str(config.spectral.wlS(1:5)', '%.1f '));
        % fprintf('  First 5 wlF: [%s]\n', num2str(config.spectral.wlF(1:5)', '%.1f '));
        % fprintf('  Spectral indices - IwlP: %s, IwlF: %s\n', ...
        %         num2str(config.spectral.IwlP(1:min(5,end))), num2str(config.spectral.IwlF(1:min(5,end))));
        
        % % Constants
        % fprintf('Constants:\n');
        % if isfield(config.constants, 'sigma')
        %     fprintf('  Stefan-Boltzmann (sigma): %.6e\n', config.constants.sigma);
        % end
        % if isfield(config.constants, 'c')
        %     fprintf('  Speed of light (c): %.6e\n', config.constants.c);
        % end
        % if isfield(config.constants, 'h')
        %     fprintf('  Planck constant (h): %.6e\n', config.constants.h);
        % end
        
        % % Angles
        % fprintf('Geometry:\n');
        % fprintf('  Sun zenith (tts): %.2f°, View zenith (tto): %.2f°, Azimuth (psi): %.2f°\n', ...
        %         params.angles.tts, params.angles.tto, params.angles.psi);
        
        % % Canopy structure
        % fprintf('Canopy:\n');
        % fprintf('  LAI: %.3f, Height: %.2f m, Layers: %d\n', ...
        %         params.canopy.LAI, params.canopy.hc, params.canopy.nlayers);
        % fprintf('  Hot spot: %.4f, Leaf width: %.3f\n', ...
        %         params.canopy.hot, params.canopy.leafwidth);
        % if isfield(params.canopy, 'lidf') && ~isempty(params.canopy.lidf)
        %     fprintf('  LIDF size: %dx%d, first few values: [%s]\n', ...
        %             size(params.canopy.lidf,1), size(params.canopy.lidf,2), ...
        %             num2str(params.canopy.lidf(1:min(3,end),1)', '%.3f '));
        % end
        
        % % Soil properties
        % fprintf('Soil:\n');
        % fprintf('  SMC: %.2f%%, Thermal reflectance: %.3f\n', ...
        %         results.soil.SMC*100, results.soil.rs_thermal);
        % if isfield(results.soil, 'refl') && ~isempty(results.soil.refl)
        %     fprintf('  Soil reflectance: %d bands, range %.3f-%.3f\n', ...
        %             length(results.soil.refl), min(results.soil.refl), max(results.soil.refl));
        %     fprintf('  First 5 soil refl: [%s]\n', num2str(results.soil.refl(1:5)', '%.3f '));
        % end
        
        % % Leaf optical properties
        % fprintf('Leaf Optics:\n');
        % if isfield(results.leafopt, 'refl') && ~isempty(results.leafopt.refl)
        %     fprintf('  Leaf reflectance: %dx%d, range %.4f-%.4f\n', ...
        %             size(results.leafopt.refl,1), size(results.leafopt.refl,2), ...
        %             min(results.leafopt.refl(:)), max(results.leafopt.refl(:)));
        % end
        % if isfield(results.leafopt, 'tran') && ~isempty(results.leafopt.tran)
        %     fprintf('  Leaf transmittance: %dx%d, range %.4f-%.4f\n', ...
        %             size(results.leafopt.tran,1), size(results.leafopt.tran,2), ...
        %             min(results.leafopt.tran(:)), max(results.leafopt.tran(:)));
        % end
        
        % % Gap fractions
        % fprintf('Gap Fractions:\n');
        % if isfield(results.gap, 'Ps') && ~isempty(results.gap.Ps)
        %     fprintf('  Sunlit fraction (Ps): %d layers, range %.4f-%.4f\n', ...
        %             length(results.gap.Ps), min(results.gap.Ps), max(results.gap.Ps));
        %     fprintf('  Mean Ps: %.4f, First 5: [%s]\n', ...
        %             mean(results.gap.Ps), num2str(results.gap.Ps(1:min(5,end))', '%.4f '));
        % end
        % if isfield(results.gap, 'Po') && ~isempty(results.gap.Po)
        %     fprintf('  Observer gap (Po): %d layers, range %.4f-%.4f\n', ...
        %             length(results.gap.Po), min(results.gap.Po), max(results.gap.Po));
        % end
        
        % % Radiation fields
        % fprintf('Radiation:\n');
        % if isfield(results.rad, 'Eout_') && ~isempty(results.rad.Eout_)
        %     fprintf('  Eout_: %d bands, range %.2f-%.2f\n', ...
        %             length(results.rad.Eout_), min(results.rad.Eout_), max(results.rad.Eout_));
        % end
        % if isfield(results.rad, 'Lo_') && ~isempty(results.rad.Lo_)
        %     fprintf('  Lo_: %d bands, range %.4f-%.4f\n', ...
        %             length(results.rad.Lo_), min(results.rad.Lo_), max(results.rad.Lo_));
        % end
        
        % % Photosynthesis efficiency (eta values)
        % fprintf('Photosynthesis Efficiency:\n');
        % if ~isempty(results.bcu.eta)
        %     fprintf('  Sunlit eta: %d layers, range %.6f-%.6f\n', ...
        %             length(results.bcu.eta), min(results.bcu.eta), max(results.bcu.eta));
        %     fprintf('  Sunlit eta mean: %.6f, std: %.6f\n', ...
        %             mean(results.bcu.eta), std(results.bcu.eta));
        %     fprintf('  First 5 sunlit eta: [%s]\n', num2str(results.bcu.eta(1:min(5,end))', '%.6f '));
            
        %     % Check for problematic eta values
        %     if any(isnan(results.bcu.eta))
        %         fprintf('  WARNING: NaN values in sunlit eta!\n');
        %     end
        %     if any(isinf(results.bcu.eta))
        %         fprintf('  WARNING: Inf values in sunlit eta!\n');
        %     end
        %     if length(unique(results.bcu.eta)) < 2
        %         fprintf('  WARNING: All sunlit eta values are identical: %.6f\n', results.bcu.eta(1));
        %     end
        % end
        
        % if ~isempty(results.bch.eta)
        %     fprintf('  Shaded eta: %d layers, range %.6f-%.6f\n', ...
        %             length(results.bch.eta), min(results.bch.eta), max(results.bch.eta));
        %     fprintf('  Shaded eta mean: %.6f, std: %.6f\n', ...
        %             mean(results.bch.eta), std(results.bch.eta));
        %     fprintf('  First 5 shaded eta: [%s]\n', num2str(results.bch.eta(1:min(5,end))', '%.6f '));
            
        %     % Check for problematic eta values
        %     if any(isnan(results.bch.eta))
        %         fprintf('  WARNING: NaN values in shaded eta!\n');
        %     end
        %     if any(isinf(results.bch.eta))
        %         fprintf('  WARNING: Inf values in shaded eta!\n');
        %     end
        %     if length(unique(results.bch.eta)) < 2
        %         fprintf('  WARNING: All shaded eta values are identical: %.6f\n', results.bch.eta(1));
        %     end
        % end
        
        % fprintf('=== End RTMf Input Debug ===\n');

        % Check for very small LAI that would cause fluorescence interpolation issues
        effective_lai = params.canopy.LAI;
        if isfield(params, 'mly') && isfield(params.mly, 'totLAI')
            effective_lai = params.mly.totLAI;
        end
        
        % Also check for NaN values in eta arrays that could cause issues
        eta_has_nan = false;
        if ~isempty(results.bcu.eta) && ~isempty(results.bch.eta)
            eta_has_nan = any(isnan(results.bcu.eta(:))) || any(isnan(results.bch.eta(:)));
        end
        
        if effective_lai < 1e-8 || eta_has_nan
            if effective_lai < 1e-8
                fprintf('INFO: Very small LAI (%.2e) detected - using placeholder fluorescence values\n', effective_lai);
            end
            if eta_has_nan
                fprintf('INFO: NaN values in fluorescence efficiency (eta) - using placeholder fluorescence values\n');
            end
            
            % Set placeholder fluorescence outputs to avoid interpolation errors
            wlF_length = length(config.spectral.wlF);
            results.rad.LoF_ = zeros(wlF_length, 1);
            results.rad.EoutF_ = zeros(wlF_length, 1);
            results.rad.LoF_sunlit = zeros(wlF_length, 1);
            results.rad.LoF_shaded = zeros(wlF_length, 1);
            results.rad.LoF_scattered = zeros(wlF_length, 1);
            results.rad.LoF_soil = zeros(wlF_length, 1);
            results.rad.EoutF = 0;
            results.rad.LoutF = 0;
            results.rad.Femleaves_ = zeros(wlF_length, 1);
            
            % Add the scalar fluorescence fields that are normally created by RTMf
            results.rad.F685 = 0;
            results.rad.wl685 = 685;  % default wavelength
            results.rad.F740 = 0;
            results.rad.wl740 = 740;  % default wavelength  
            results.rad.F684 = 0;
            results.rad.F761 = 0;
            
            % Add fields normally created in calculate_derived_products
            results.rad.PoutFrc = 0;
            results.rad.EoutFrc_ = zeros(wlF_length, 1);
            results.rad.EoutFrc = 0;
            results.rad.sigmaF = zeros(wlF_length, 1);
        else
            results.rad = RTMf(config.constants, config.spectral, results.rad, results.soil, ...
                        results.leafopt, params.canopy, results.gap, params.angles, ...
                        results.bcu.eta, results.bch.eta);
        end
    end
                    
    %     try
    %         fprintf('=== Starting RTMf calculation ===\n');

    %         fprintf('RTMf completed successfully\n');
    %         % % print out all variables in results.rad
    %         % fprintf('=== RTMf Output Variables ===\n');
    %         % var_names = fieldnames(results.rad);
    %         % for i = 1:length(var_names)
    %         %     var_name = var_names{i};
    %         %     if isnumeric(results.rad.(var_name))
    %         %         fprintf('  %s: size [%s], range %.6f to %.6f\n', ...
    %         %             var_name, num2str(size(results.rad.(var_name))), ...
    %         %             min(results.rad.(var_name)(:)), max(results.rad.(var_name)(:)));
    %         %     end
    %         % end
    %         % % Debug RTMf outputs
    %         % fprintf('=== RTMf Output Debug ===\n');
    %         % if isfield(results.rad, 'LoF_')
    %         %     fprintf('  LoF_ size: [%s], range: %.6f to %.6f\n', ...
    %         %         num2str(size(results.rad.LoF_)), min(results.rad.LoF_(:)), max(results.rad.LoF_(:)));
    %         %     fprintf('  NaN count in LoF_: %d/%d\n', sum(isnan(results.rad.LoF_(:))), numel(results.rad.LoF_));
    %         %     if all(isnan(results.rad.LoF_(:)))
    %         %         fprintf('  ERROR: All LoF_ values are NaN!\n');
    %         %     elseif any(isnan(results.rad.LoF_(:)))
    %         %         fprintf('  WARNING: Some LoF_ values are NaN\n');
    %         %     end
    %         % else
    %         %     fprintf('  ERROR: LoF_ field not found in RTMf output\n');
    %         % end
            
    %     catch ME
    %         fprintf('RTMf failed with error: %s\n', ME.message);
    %         fprintf('Error location: %s\n', ME.stack(1).name);
            
    %         % Additional debugging for interpolation errors
    %         if contains(ME.message, 'Interpolation requires at least two sample points')
    %             fprintf('=== Interpolation Error Debug ===\n');
                
    %             % Check spectral data more thoroughly
    %             if isfield(config.spectral, 'wlF') && isfield(config.spectral, 'wlS')
    %                 fprintf('Spectral bands - Solar: %d, Fluor: %d\n', ...
    %                     length(config.spectral.wlS), length(config.spectral.wlF));
                    
    %                 % Check for duplicate wavelengths
    %                 if length(unique(config.spectral.wlF)) < length(config.spectral.wlF)
    %                     warning('Duplicate wavelengths found in fluorescence bands');
    %                 end
    %             end
                
    %             % Check if eta values are all identical (could cause interpolation issues)
    %             if length(unique(results.bcu.eta)) < 2
    %                 warning('All sunlit eta values are identical: %.6f', results.bcu.eta(1));
    %             end
    %             if length(unique(results.bch.eta)) < 2
    %                 warning('All shaded eta values are identical: %.6f', results.bch.eta(1));
    %             end
    %         end
            
    %         rethrow(ME);
    %     end
    % end
    
    % Xanthophyll cycle calculation
    if config.options.calc_xanthophyllabs
        results.rad = RTMz(config.constants, config.spectral, results.rad, results.soil, ...
                          results.leafopt, params.canopy, results.gap, params.angles, ...
                          results.bcu.Kn, results.bch.Kn);
    end
    
    % Thermal radiative transfer
    results.rad = RTMt_sb(config.constants, results.rad, results.soil, params.leafbio, params.canopy, ...
                         results.gap, results.thermal.Tcu, results.thermal.Tch, results.thermal.Tsu, ...
                         results.thermal.Tsh, 1, config.spectral);
    
    % Planck emission calculation
    if config.options.calc_planck
        results.rad = RTMt_planck(config.spectral, results.rad, results.soil, results.leafopt, ...
                                 params.canopy, results.gap, results.thermal.Tcu, results.thermal.Tch, ...
                                 results.thermal.Tsu, results.thermal.Tsh);
    end
end

function results = calculate_derived_products(params, config, results)
% Calculate derived canopy products exactly as in SCOPE.m
    
    % Extract variables to match SCOPE.m naming
    canopy = params.canopy;
    gap = results.gap;
    rad = results.rad;
    bch = results.bch;
    bcu = results.bcu;
    leafbio = params.leafbio;
    options = config.options;
    spectral = config.spectral;
    constants = config.constants;
    optipar = config.optipar;
    integr = config.integr;
    
    nl = canopy.nlayers;
    
    % Exact calculations from SCOPE.m
    Ps = gap.Ps(1:nl);
    Ph = (1-Ps);

    canopy.LAIsunlit = canopy.LAI*mean(Ps);
    canopy.LAIshaded = canopy.LAI-canopy.LAIsunlit;

    canopy.Pnsun_Cab = canopy.LAI*meanleaf(canopy,rad.Pnu_Cab,integr,Ps); % net PAR Cab sunlit leaves (photons)
    canopy.Pnsha_Cab = canopy.LAI*meanleaf(canopy,rad.Pnh_Cab,'layers',Ph); % net PAR Cab shaded leaves (photons)
    canopy.Pntot_Cab = canopy.Pnsun_Cab+canopy.Pnsha_Cab; % net PAR Cab leaves (photons)

    canopy.Pnsun_Car = canopy.LAI*meanleaf(canopy,rad.Pnu_Car,integr,Ps); % net PAR Cab sunlit leaves (photons)
    canopy.Pnsha_Car = canopy.LAI*meanleaf(canopy,rad.Pnh_Car,'layers',Ph); % net PAR Cab shaded leaves (photons)
    canopy.Pntot_Car = canopy.Pnsun_Car+canopy.Pnsha_Car; % net PAR Cab leaves (photons)

    canopy.Pnsun = canopy.LAI*meanleaf(canopy,rad.Pnu,integr,Ps); % net PAR sunlit leaves (photons)
    canopy.Pnsha = canopy.LAI*meanleaf(canopy,rad.Pnh,'layers',Ph); % net PAR shaded leaves (photons)
    canopy.Pntot = canopy.Pnsun+canopy.Pnsha; % net PAR leaves (photons)

    canopy.Rnsun_Cab = canopy.LAI*meanleaf(canopy,rad.Rnu_Cab,integr,Ps); % net PAR Cab sunlit leaves (radiance)
    canopy.Rnsha_Cab = canopy.LAI*meanleaf(canopy,rad.Rnh_Cab,'layers',Ph); % net PAR Cab sunlit leaves (radiance)
    canopy.Rntot_Cab = canopy.Rnsun_Cab+canopy.Rnsha_Cab; % net PAR Cab leaves (radiance)

    canopy.Rnsun_Car = canopy.LAI*meanleaf(canopy,rad.Rnu_Car,integr,Ps); % net PAR Cab sunlit leaves (radiance)
    canopy.Rnsha_Car = canopy.LAI*meanleaf(canopy,rad.Rnh_Car,'layers',Ph); % net PAR Cab sunlit leaves (radiance)
    canopy.Rntot_Car = canopy.Rnsun_Car+canopy.Rnsha_Car; % net PAR Cab leaves (radiance)

    canopy.Rnsun_PAR = canopy.LAI*meanleaf(canopy,rad.Rnu_PAR,integr,Ps); % net PAR sunlit leaves (radiance)
    canopy.Rnsha_PAR = canopy.LAI*meanleaf(canopy,rad.Rnh_PAR,'layers',Ph); % net PAR sunlit leaves (radiance)
    canopy.Rntot_PAR = canopy.Rnsun_PAR+canopy.Rnsha_PAR; % net PAR leaves (radiance)

    % LST [K] (directional, but assuming black-body surface!)
    canopy.LST      = (pi*(rad.Lot+rad.Lote)./(constants.sigmaSB*rad.canopyemis)).^0.25;
    canopy.emis     = rad.canopyemis;

    % photosynthesis [mumol CO2 m-2 s-1]
    canopy.A        = canopy.LAI*(meanleaf(canopy,bch.A,'layers',Ph)+meanleaf(canopy,bcu.A,integr,Ps)); % net photosynthesis of leaves
    canopy.GPP      = canopy.LAI*(meanleaf(canopy,bch.Ag,'layers',Ph)+meanleaf(canopy,bcu.Ag,integr,Ps)); % gross photosynthesis

    % electron transport rate [mumol m-2 s-1]
    canopy.Ja       = canopy.LAI*(meanleaf(canopy,bch.Ja,'layers',Ph)+meanleaf(canopy,bcu.Ja,integr,Ps)); % electron transport

    % non-photochemical quenching (energy) [W m-2]
    canopy.ENPQ     = canopy.LAI*(meanleaf(canopy,rad.Rnh_Cab.*bch.Phi_N,'layers',Ph)+meanleaf(canopy,rad.Rnu_Cab.*bcu.Phi_N,integr,Ps)); % NPQ energy;
    canopy.PNPQ     = canopy.LAI*(meanleaf(canopy,rad.Pnh_Cab.*bch.Phi_N,'layers',Ph)+meanleaf(canopy,rad.Pnu_Cab.*bcu.Phi_N,integr,Ps)); % NPQ energy;

    % computation of re-absorption corrected fluorescence
    % Yang and Van der Tol (2019); Van der Tol et al. (2019)
    %aPAR_Cab_eta    = canopy.LAI*(meanleaf(canopy,bch.eta .* rad.Rnh_Cab,'layers',Ph)+meanleaf(canopy,bcu.eta .* rad.Rnu_Cab,integr,Ps)); %
    aPAR_Cab_eta    = canopy.LAI*(meanleaf(canopy,bch.eta .* rad.Pnh_Cab,'layers',Ph)+meanleaf(canopy,bcu.eta .* rad.Pnu_Cab,integr,Ps)); %

    if options.calc_fluor
        ep              = constants.A*ephoton(spectral.wlF'*1E-9,constants);
        rad.PoutFrc     = leafbio.fqe*aPAR_Cab_eta;
        rad.EoutFrc_    = 1E-3*ep.*(rad.PoutFrc*optipar.phi(spectral.IwlF)); %1E-6: umol2mol, 1E3: nm-1 to um-1
        rad.EoutFrc     = 1E-3*Sint(rad.EoutFrc_,spectral.wlF);
        sigmaF          = pi*rad.LoF_./rad.EoutFrc_;
        rad.sigmaF      = interp1(spectral.wlF(1:4:end),sigmaF(1:4:end),spectral.wlF);
        canopy.fqe      = rad.PoutFrc./canopy.Pntot_Cab;
    else
        canopy.fqe = nan;
    end

    rad.Lotot_      = rad.Lo_+rad.Lot_;
    rad.Eout_       = rad.Eout_+rad.Eoutte_;
    if options.calc_fluor
        rad.Lototf_     = rad.Lotot_;
        rad.Lototf_(spectral.IwlF') = rad.Lototf_(spectral.IwlF)+rad.LoF_;
        rad.reflapp = rad.refl;
        rad.reflapp(spectral.IwlF) =pi*rad.Lototf_(spectral.IwlF)./(rad.Esun_(spectral.IwlF)+rad.Esky_(spectral.IwlF));
    end

    % Note: directional calculations and calc_brdf not implemented in grid version
    % if options.calc_directional
    %     directional = calc_brdf(constants,options,directional,spectral,angles,atmo,soil,leafopt,canopy,meteo,thermal,bcu,bch);
    %     savebrdfoutput(options,directional,angles,spectral,Output_dir)
    % end

    rad.Lo = 0.001 * Sint(rad.Lo_(spectral.IwlP),spectral.wlP);
    
    % Store results back
    results.canopy = canopy;
    results.rad = rad;
    
end

% %% ========================================================================
% %% RESULT ACCUMULATION AND OUTPUT
% %% ========================================================================

function acc = initialize_accumulator(ny, nx, nt, config)
% Initialize result accumulator structure based on first pixel results
    
    acc.meta.ny = ny; 
    acc.meta.nx = nx; 
    acc.meta.nt = nt;
    acc.meta.wlS = config.spectral.wlS(:); 
    acc.meta.wlF = config.spectral.wlF(:);
    acc.meta.hasFluo  = isfield(config.options,'calc_fluor') && config.options.calc_fluor==1;
    acc.meta.hasSpect = isfield(config.options,'save_spectral') && config.options.save_spectral==1;

    % Calculate estimated memory usage before initializing arrays
    fprintf('Calculating estimated memory requirements...\n');
    
    % Basic dimensions
    bytes_per_single = 4; % single precision floating point
    total_pixels = ny * nx * nt;
    
    % Display key dimensions
    fprintf('  Grid dimensions: %d x %d x %d (ny x nx x nt)\n', ny, nx, nt);
    fprintf('  Total pixels to process: %d\n', total_pixels);
    
    % Calculate memory for basic data arrays
    coord_mem = 2 * nt * bytes_per_single; % year, doy
    apar_mem = 19 * total_pixels * bytes_per_single; % 19 apar variables
    veg_mem = 8 * total_pixels * bytes_per_single; % 8 vegetation variables
    radS_mem = 7 * total_pixels * bytes_per_single; % 7 radiation variables
    resist_mem = 4 * total_pixels * bytes_per_single; % 4 resistance variables
    
    basic_mem = coord_mem + apar_mem + veg_mem + radS_mem + resist_mem;
    total_mem = basic_mem;
    
    fprintf('  Basic arrays: %.1f MB\n', basic_mem / 1e6);
    
    % Add fluorescence memory if enabled
    if acc.meta.hasFluo
        nF = numel(acc.meta.wlF);
        nS = numel(acc.meta.wlS);
        fprintf('  Fluorescence enabled: %d fluorescence wavelengths, %d solar wavelengths\n', nF, nS);
        
        fluor_scalar_mem = 9 * total_pixels * bytes_per_single; % 9 scalar fluorescence variables
        fluor_specF_mem = 5 * nF * total_pixels * bytes_per_single; % 5 spectral fluorescence variables
        radS_wlS_mem = 2 * nS * total_pixels * bytes_per_single; % 2 spectral radiation variables
        
        fluor_total_mem = fluor_scalar_mem + fluor_specF_mem + radS_wlS_mem;
        total_mem = total_mem + fluor_total_mem;
        
        fprintf('    Scalar fluorescence: %.1f MB\n', fluor_scalar_mem / 1e6);
        fprintf('    Spectral fluorescence (%d wl): %.1f MB\n', nF, fluor_specF_mem / 1e6);
        fprintf('    Solar spectral radiation (%d wl): %.1f MB\n', nS, radS_wlS_mem / 1e6);
        fprintf('  Total fluorescence arrays: %.1f MB\n', fluor_total_mem / 1e6);
    end
    
    % Add spectral memory if enabled
    if acc.meta.hasSpect
        nS = numel(acc.meta.wlS);
        fprintf('  Spectral output enabled: %d wavelengths\n', nS);
        
        specS_mem = 9 * nS * total_pixels * bytes_per_single; % 9 spectral variables
        total_mem = total_mem + specS_mem;
        
        fprintf('  Spectral arrays (%d wl × 9 vars): %.1f MB\n', nS, specS_mem / 1e6);
        
        % Memory breakdown per wavelength for spectral data
        mem_per_wl = 9 * total_pixels * bytes_per_single / 1e6;
        fprintf('    Memory per wavelength: %.2f MB\n', mem_per_wl);
    end
    
    % Add estimated flux memory (will be updated after first pixel)
    estimated_flux_vars = 10; % Conservative estimate
    flux_mem = estimated_flux_vars * total_pixels * bytes_per_single;
    total_mem = total_mem + flux_mem;
    fprintf('  Flux arrays (estimated %d vars): %.1f MB\n', estimated_flux_vars, flux_mem / 1e6);
    
    % Display total memory summary
    fprintf('  ------------------------\n');
    fprintf('  Total estimated memory: %.1f MB (%.2f GB)\n', total_mem / 1e6, total_mem / 1e9);
    
    % Check available memory and provide warnings/suggestions
    try
        [~, sys_view] = memory;
        available_gb = sys_view.PhysicalMemory.Available / 1e9;
        total_gb = sys_view.PhysicalMemory.Total / 1e9;
        fprintf('  Available system memory: %.2f GB (%.2f GB total)\n', available_gb, total_gb);
        
        memory_ratio = total_mem / 1e9 / available_gb;
        if memory_ratio > 1.0
            fprintf('  *** WARNING: Estimated memory (%.2f GB) EXCEEDS available memory! ***\n', total_mem / 1e9);
            fprintf('  *** Processing will likely fail or cause system instability! ***\n');
        elseif memory_ratio > 0.8
            fprintf('  *** WARNING: High memory usage (%.1f%% of available) ***\n', memory_ratio * 100);
        elseif memory_ratio > 0.5
            fprintf('  ** Moderate memory usage (%.1f%% of available) **\n', memory_ratio * 100);
        else
            fprintf('  Memory usage looks reasonable (%.1f%% of available)\n', memory_ratio * 100);
        end
    catch
        fprintf('  (Unable to check available system memory)\n');
        if total_mem / 1e9 > 16 % Arbitrary threshold for very high memory
            fprintf('  *** WARNING: Very high memory usage (%.2f GB) ***\n', total_mem / 1e9);
        end
    end
    
    % Provide memory reduction suggestions if memory usage is high
    if total_mem / 1e9 > 32 % More than 32 GB
        fprintf('\n  MEMORY REDUCTION SUGGESTIONS:\n');
        fprintf('  - Consider processing smaller spatial tiles\n');
        fprintf('  - Reduce temporal resolution if possible\n');
        if acc.meta.hasSpect && exist('nS', 'var')
            fprintf('  - Consider disabling spectral output (save_spectral=0) to save %.1f GB\n', specS_mem / 1e9);
            fprintf('  - Or reduce spectral resolution (currently %d wavelengths)\n', nS);
        end
        if acc.meta.hasFluo
            fprintf('  - Consider disabling fluorescence (calc_fluor=0) to save %.1f GB\n', fluor_total_mem / 1e9);
        end
        fprintf('  - Process data in temporal chunks rather than all at once\n');
    end
    fprintf('\n');

    acc.coord.year = nan(1,nt,'single');
    acc.coord.doy  = nan(1,nt,'single');

    % Initialize data arrays for different output categories
    acc.apar.names = { ...
        'rad_PAR','rad_EPAR', ...
        'LAIsunlit','LAIshaded', ...
        'Pntot','Pnsun','Pnsha', ...
        'Pntot_Cab','Pnsun_Cab','Pnsha_Cab', ...
        'Pntot_Car','Pnsun_Car','Pnsha_Car', ...
        'Rntot_PAR','Rnsun_PAR','Rnsha_PAR', ...
        'Rntot_Cab','Rnsun_Cab','Rnsha_Cab', ...
        'Rntot_Car','Rnsun_Car','Rnsha_Car' };
    acc.apar.data = nan(numel(acc.apar.names), ny, nx, nt, 'single');

    acc.veg.names = {'A','Ja','ENPQ','PNPQ','fqe','LST','emis','GPP'};
    acc.veg.data  = nan(numel(acc.veg.names), ny, nx, nt, 'single');

    acc.radS.names = {'meteo_Rin','meteo_Rli','Eouto','Eoutt_plus_Eoutte','Lo_scalar','Lot_scalar','Lote_scalar'};
    acc.radS.data  = nan(numel(acc.radS.names), ny, nx, nt, 'single');

    acc.resist.names = {'raa','raws','rss','ustar'};
    acc.resist.data  = nan(numel(acc.resist.names), ny, nx, nt, 'single');

    if acc.meta.hasFluo
        % Scalar fluorescence data
        acc.fluor.names_scalar = {'F685','wl685','F740','wl740','F684','F761','LoutF','EoutF','EoutFrc'};
        acc.fluor.scalar = nan(numel(acc.fluor.names_scalar), ny, nx, nt, 'single');

        % Spectral fluorescence data - PRE-ALLOCATED ARRAYS
        nF = numel(acc.meta.wlF);
        acc.fluor.names_specF = {'LoF','sigmaF','EoutFrc_','Femleaves','EoutF_hemis'};
        acc.fluor.specF = nan(numel(acc.fluor.names_specF), nF, ny, nx, nt, 'single');
        
        % Solar spectral radiation data - PRE-ALLOCATED ARRAYS  
        nS = numel(acc.meta.wlS);
        acc.radS_wlS.names = {'Lototf','reflapp'};
        acc.radS_wlS.data = nan(numel(acc.radS_wlS.names), nS, ny, nx, nt, 'single');
        
        fprintf('  INFO: Pre-allocated fluorescence arrays - efficient direct assignment!\n');
    else
        acc.fluor = struct();
        acc.radS_wlS = struct();
    end

    if acc.meta.hasSpect
        % Spectral data - PRE-ALLOCATED ARRAYS
        nS = numel(acc.meta.wlS);
        acc.specS.names = {'refl','rsd','rdd','rso','rdo','Eout','Lotot','Esun','Esky'};
        acc.specS.data = nan(numel(acc.specS.names), nS, ny, nx, nt, 'single');
        
        fprintf('  INFO: Pre-allocated spectral arrays - efficient direct assignment!\n');
    else
        acc.specS = struct();
    end

    acc.flux.names = {'Rnctot','lEctot','Hctot','Actot','Tcave', 'Rnstot','lEstot','Hstot','Gtot','Tsave', 'Rntot','lEtot','Htot'};
    acc.flux.data  = nan(numel(acc.flux.names), ny, nx, nt, 'single');

    acc.pars = struct('len',0);  % No parameter tracking for now
end


function acc = accumulate_results(acc, pixel_results, iy, ix, t)
% Accumulate results from a single pixel into the accumulator
% Improved version with direct array assignment for spectral data
    
    xyt = pixel_results.xyt;
    rad = pixel_results.rad;
    canopy = pixel_results.canopy;
    fluxes = pixel_results.fluxes;
    meteo = pixel_results.meteo;
    resistance = pixel_results.resistance;
    
    % Store time coordinates
    [doy, yr] = local_doy_year(xyt, t);
    acc.coord.doy(1,t)  = single(doy);
    acc.coord.year(1,t) = single(yr);

    % Accumulate APAR data
    Ablock = [
        rad.PAR; rad.EPAR; canopy.LAIsunlit; canopy.LAIshaded;
        canopy.Pntot; canopy.Pnsun; canopy.Pnsha;
        canopy.Pntot_Cab; canopy.Pnsun_Cab; canopy.Pnsha_Cab;
        canopy.Pntot_Car; canopy.Pnsun_Car; canopy.Pnsha_Car;
        canopy.Rntot_PAR; canopy.Rnsun_PAR; canopy.Rnsha_PAR;
        canopy.Rntot_Cab; canopy.Rnsun_Cab; canopy.Rnsha_Cab;
        canopy.Rntot_Car; canopy.Rnsun_Car; canopy.Rnsha_Car];
    acc.apar.data(:,iy,ix,t) = single(Ablock);

    % Accumulate vegetation data
    Vblock = [canopy.A; canopy.Ja; canopy.ENPQ; canopy.PNPQ;
              canopy.fqe; canopy.LST; canopy.emis; canopy.GPP];
    acc.veg.data(:,iy,ix,t) = single(Vblock);

    % Accumulate radiation data
    Rblock = [meteo.Rin; meteo.Rli; rad.Eouto; rad.Eoutt + rad.Eoutte;
              rad.Lo; rad.Lot; rad.Lote];
    acc.radS.data(:,iy,ix,t) = single(Rblock);

    % Accumulate resistance data
    acc.resist.data(:,iy,ix,t) = single([resistance.raa; resistance.raws; 
                                        resistance.rss; resistance.ustar]);

    % Accumulate fluorescence data (if enabled) - DIRECT ARRAY ASSIGNMENT
    if isfield(acc,'fluor') && ~isempty(fieldnames(acc.fluor))
        % Extract fluorescence scalar fields
        Fsc = [rad.F685; rad.wl685; rad.F740; rad.wl740; 
               rad.F684; rad.F761; rad.LoutF; rad.EoutF; rad.EoutFrc];
        
        acc.fluor.scalar(:,iy,ix,t) = single(Fsc);

        % Direct assignment for fluorescence spectral data
        if isfield(acc.fluor, 'specF')
            specF_fields = {'LoF_', 'sigmaF', 'EoutFrc_', 'Femleaves_', 'EoutF_'};
            
            for iVar = 1:5
                acc.fluor.specF(iVar,:,iy,ix,t) = single(rad.(specF_fields{iVar})(:));
            end
        end

        % Direct assignment for solar spectral data
        if isfield(acc,'radS_wlS') && isfield(acc.radS_wlS,'data')
            radS_fields = {'Lototf_', 'reflapp'};
            
            for iVar = 1:2
                acc.radS_wlS.data(iVar,:,iy,ix,t) = single(rad.(radS_fields{iVar})(:));
            end
        end
    end

    % Direct assignment for spectral data - MUCH MORE EFFICIENT
    if isfield(acc,'specS') && isfield(acc.specS,'data')
        specS_fields = {'refl', 'rsd', 'rdd', 'rso', 'rdo', 'Eout_', 'Lotot_', 'Esun_', 'Esky_'};
        
        for iVar = 1:9
            acc.specS.data(iVar,:,iy,ix,t) = single(rad.(specS_fields{iVar})(:));
        end
    end

    % accumulate flux data
    Fblock = [fluxes.Rnctot; fluxes.lEctot; fluxes.Hctot; fluxes.Actot; fluxes.Tcave; 
              fluxes.Rnstot; fluxes.lEstot; fluxes.Hstot; fluxes.Gtot; fluxes.Tsave; 
              fluxes.Rntot; fluxes.lEtot; fluxes.Htot];
    acc.flux.data(:,iy,ix,t) = single(Fblock);

end


function write_results_to_netcdf(results, out_nc, ~)
% Write accumulated results to NetCDF file
    
    acc = results;
    ny = acc.meta.ny; nx = acc.meta.nx; nt = acc.meta.nt;
    
    % Delete existing file
    if exist(out_nc,'file'), delete(out_nc); end

    % Create coordinate variables
    nccreate(out_nc,'time','Dimensions',{'time',nt},'Datatype','single');
    nccreate(out_nc,'y','Dimensions',{'y',ny},'Datatype','int32');
    nccreate(out_nc,'x','Dimensions',{'x',nx},'Datatype','int32');
    ncwrite(out_nc,'time',single(1:nt)); 
    ncwrite(out_nc,'y',int32(1:ny)); 
    ncwrite(out_nc,'x',int32(1:nx));

    % Create time coordinate data
    nccreate(out_nc,'year','Dimensions',{'time',nt},'Datatype','int32');
    nccreate(out_nc,'doy','Dimensions',{'time',nt},'Datatype','single');
    ncwrite(out_nc,'year',int32(acc.coord.year));
    ncwrite(out_nc,'doy', single(acc.coord.doy));

    % Create wavelength coordinates
    if ~isempty(acc.meta.wlS)
        nccreate(out_nc,'wlS','Dimensions',{'wlS',numel(acc.meta.wlS)},'Datatype','single');
        ncwrite(out_nc,'wlS',single(acc.meta.wlS));
    end
    if isfield(acc.meta,'hasFluo') && acc.meta.hasFluo && ~isempty(acc.meta.wlF)
        nccreate(out_nc,'wlF','Dimensions',{'wlF',numel(acc.meta.wlF)},'Datatype','single');
        ncwrite(out_nc,'wlF',single(acc.meta.wlF));
    end

    % Write main data blocks
    write4_block(out_nc,'apar',       acc.apar.data,    acc.apar.names);
    write4_block(out_nc,'veg',        acc.veg.data,     acc.veg.names);
    write4_block(out_nc,'rad_scalar', acc.radS.data,    acc.radS.names);
    write4_block(out_nc,'resistance', acc.resist.data,  acc.resist.names);

    % Write fluorescence data (if present)
    if isfield(acc,'fluor') && ~isempty(fieldnames(acc.fluor))
        write4_block(out_nc,'fluor_scalar', acc.fluor.scalar, acc.fluor.names_scalar);
        if isfield(acc.fluor,'specF')
            write5_block(out_nc,'fluor_specF', acc.fluor.specF, acc.fluor.names_specF, 'wlF');
        end
        if isfield(acc,'radS_wlS') && isfield(acc.radS_wlS,'data')
            write5_block(out_nc,'rad_withF_wlS', acc.radS_wlS.data, acc.radS_wlS.names, 'wlS');
        end
    end

    % Write spectral data (if present)
    if isfield(acc,'specS') && isfield(acc.specS,'data')
        write5_block(out_nc,'specS', acc.specS.data, acc.specS.names, 'wlS');
    end

    % Write flux data
    if ~isempty(acc.flux.data)
        nccreate(out_nc,'fluxes','Dimensions',{'flux', size(acc.flux.data,1), 'time', nt, 'y', ny, 'x', nx}, 'Datatype','single','DeflateLevel',4);
        ncwrite(out_nc,'fluxes', permute(acc.flux.data, [1 4 2 3]));
        ncwriteatt(out_nc,'fluxes','names', strjoin(acc.flux.names, ','));
    end

    % Write parameter data (if present)
    if isfield(acc,'pars') && isfield(acc.pars,'len') && acc.pars.len>0
        nccreate(out_nc,'pars','Dimensions',{'par', acc.pars.len, 'time', nt, 'y', ny, 'x', nx}, 'Datatype','single','DeflateLevel',4);
        ncwrite(out_nc,'pars', permute(acc.pars.data, [1 4 2 3]));
        ncwriteatt(out_nc,'pars','desc','[count, then values of variables with vmax>1]');
    end

    % Write global attributes
    ncwriteatt(out_nc,'/','title','SCOPE outputs (block-structured, refactored)');
    ncwriteatt(out_nc,'/','history',datestr(now));
    
    fprintf('Wrote NetCDF: %s\n', out_nc);
end

function write4_block(out_nc, varname, A, names)
% Write 4D data block: [var,y,x,t] -> [var,t,y,x]
    [nVar,ny,nx,nt] = size(A);
    var_dim_name = sprintf('var_%s', varname);  % Create unique dimension name
    
    % Create coordinate variable for the variable dimension
    nccreate(out_nc, var_dim_name, 'Dimensions', {var_dim_name, nVar}, 'Datatype', 'int32');
    ncwrite(out_nc, var_dim_name, int32(1:nVar));

    % Create the main data variable
    nccreate(out_nc, varname, 'Dimensions',{var_dim_name,nVar,'time',nt,'y',ny,'x',nx}, 'Datatype','single','DeflateLevel',4);
    ncwrite(out_nc, varname, permute(A, [1 4 2 3]));
    ncwriteatt(out_nc,varname,'names', strjoin(names, ','));
end

function write5_block(out_nc, varname, A, names, wlname)
% Write 5D spectral data block: [var,wl,y,x,t] -> [var,wl,t,y,x]
    [nVar,nwl,ny,nx,nt] = size(A);
    var_dim_name = sprintf('var_%s', varname);  % Create unique dimension name
    
    % Debug the data being written
    if strcmp(varname, 'fluor_specF')
        fprintf('DEBUG write5_block: Writing %s with size [%s]\n', varname, num2str(size(A)));
        fprintf('DEBUG write5_block: Data range: %.6f to %.6f\n', min(A(:)), max(A(:)));
        fprintf('DEBUG write5_block: NaN count: %d/%d\n', sum(isnan(A(:))), numel(A));
        
        % Check the first variable (LoF)
        lof_data = squeeze(A(1,:,:,:,:));
        fprintf('DEBUG write5_block: LoF data size: [%s], range: %.6f to %.6f\n', ...
            num2str(size(lof_data)), min(lof_data(:)), max(lof_data(:)));
        fprintf('DEBUG write5_block: LoF NaN count: %d/%d\n', sum(isnan(lof_data(:))), numel(lof_data));
    end
    
    % Create coordinate variable for the variable dimension
    nccreate(out_nc, var_dim_name, 'Dimensions', {var_dim_name, nVar}, 'Datatype', 'int32');
    ncwrite(out_nc, var_dim_name, int32(1:nVar));
    
    % Create the main data variable
    nccreate(out_nc, varname, 'Dimensions',{var_dim_name,nVar, wlname, nwl, 'time', nt, 'y', ny, 'x', nx}, 'Datatype','single','DeflateLevel',4);
    ncwrite(out_nc, varname, permute(A, [1 2 5 3 4]));
    ncwriteatt(out_nc,varname,'names', strjoin(names, ','));
end

%% ========================================================================
%% UTILITY FUNCTIONS (unchanged from original)
%% ========================================================================

function [ny,nx,nt] = get_dims(I)
    ny = 1; nx = 1; nt = 1;
    for i = 1:length(I.Dimensions)
        switch lower(I.Dimensions(i).Name)
            case {'y'}, ny = I.Dimensions(i).Length;
            case {'x'}, nx = I.Dimensions(i).Length;
            case {'time','t'}, nt = I.Dimensions(i).Length;
        end
    end
end

function var = read_nc_var(filename, I, aliases, ny, nx, nt, default_val)
    var = [];
    for alias = aliases
        for j = 1:length(I.Variables)
            if strcmpi(I.Variables(j).Name, alias{1})
                var = ncread(filename, I.Variables(j).Name);
                return;
            end
        end
    end
    if isempty(var)
        if isnan(default_val)
            error('Required variable %s not found in NC', aliases{1});
        end
        var = repmat(default_val, [nt,ny,nx]);
    end
end

function val = pick1(var, iy, ix, t, default_val)
    if numel(var) == 1
        val = var;
    elseif ndims(var) == 3
        val = var(t,iy,ix);
    elseif ndims(var) == 2
        val = var(iy,ix);
    else
        val = default_val;
    end
    if isnan(val), val = default_val; end
end

function [optipar_file, soil_file, atmos_file, lidf_file, opts] = read_global_opts(I)
    optipar_file=''; soil_file=''; atmos_file=''; lidf_file='';
    % SCOPE default options (matching input_data_default.csv behavior)
    opts = struct('lite',0,'calc_fluor',0,'calc_planck',0,'calc_xanthophyllabs',0, ...
                  'soilspectrum',0,'save_spectral',0,'MoninObukhov',1,'soil_heat_method',0, ...
                  'verify',0,'saveCSV',0,'mSCOPE',0,'calc_directional',0,'calc_vert_profiles',0, ...
                  'calc_rss_rbs',1,'Fluorescence_model',0,'apply_T_corr',1,'Cca_function_of_Cab',0);
    for a=1:numel(I.Attributes)
        key = lower(I.Attributes(a).Name); val = I.Attributes(a).Value;
        switch key
            case 'optipar_file',  optipar_file = char(val);
            case 'soil_file',     soil_file    = char(val);
            case 'atmos_file',    atmos_file   = char(val);
            case 'lidf_file',     lidf_file    = char(val);
            case 'soilspectrum',  opts.soilspectrum = double(val);
            case 'calc_fluor',    opts.calc_fluor   = double(val);
            case 'calc_planck',   opts.calc_planck  = double(val);
            case 'calc_xanthophyllabs', opts.calc_xanthophyllabs = double(val);
            case 'save_spectral', opts.save_spectral = double(val);
            case 'lite',          opts.lite = double(val);
            case 'moninobukhov',  opts.MoninObukhov = double(val);
            case 'soil_heat_method', opts.soil_heat_method = double(val);
        end
    end
end

function s = iff(cond,a,b), if cond, s=a; else, s=b; end, end

function [doy_k, year_k] = local_doy_year(xyt, k)
    if isfield(xyt,'t') && isa(xyt.t,'datetime')
        get_doy = @(x) juliandate(x) - juliandate(datetime(year(x),1,0));
        doy_k  = get_doy(xyt.t(k));
        year_k = year(xyt.t(k));
    else
        if isfield(xyt,'t') && numel(xyt.t)>=k, doy_k = double(xyt.t(k)); else, doy_k = k; end
        if isfield(xyt,'year') && numel(xyt.year)>=k, year_k = double(xyt.year(k)); else, year_k = 2000; end
    end
end

function atmo = load_atmo_placeholder(spectral)
    atmo.Esun_ = ones(size(spectral.wlS))*1000;
    atmo.Esky_ = ones(size(spectral.wlS))*10;
    atmo.Esun  = 1000; 
    atmo.Esky = 10;
end

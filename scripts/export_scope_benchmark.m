function export_scope_benchmark(output_path, case_index)
%EXPORT_SCOPE_BENCHMARK Export one upstream SCOPE scene as a MATLAB parity fixture.
%
% This runs a single SCOPE scene through the upstream MATLAB kernels and
% writes a flat MAT file with the exact inputs and outputs needed by the
% Python parity harness.

if nargin < 1 || isempty(output_path)
    output_path = fullfile(default_repo_root(), 'tests', 'data', 'scope_case_001.mat');
end
if nargin < 2 || isempty(case_index)
    case_index = 1;
end

repo_root = default_repo_root();
scope_root = fullfile(repo_root, 'upstream', 'SCOPE');

addpath(fullfile(scope_root, 'src', 'RTMs'));
addpath(fullfile(scope_root, 'src', 'supporting'));
addpath(fullfile(scope_root, 'src', 'fluxes'));
addpath(fullfile(scope_root, 'src', 'IO'));

cwd = pwd;
cleanup = onCleanup(@() cd(cwd));
cd(scope_root);

constants = define_constants();
path_input = 'input/';

[parameter_files, options] = load_options(path_input);
options.verify = 0;
options.saveCSV = 0;
options.calc_directional = 0;
options.calc_vert_profiles = 0;
options.calc_xanthophyllabs = 0;

if options.simulation ~= 0
    error('export_scope_benchmark currently expects simulation == 0, got %d', options.simulation);
end

if options.lite == 0
    integr = 'angles_and_layers';
else
    integr = 'layers';
end

[F, V, options] = load_inputs(path_input, parameter_files, options);

load(fullfile(path_input, 'fluspect_parameters', F(3).FileName), 'optipar');
if options.soilspectrum == 0
    rsfile = load(fullfile(path_input, 'soil_spectra', F(2).FileName));
else
    rsfile = [];
end

canopy = struct;
canopy.nlincl = 13;
canopy.nlazi = 36;
canopy.litab = [5:10:75 81:2:89]';
canopy.lazitab = 5:10:355;
soilemp.SMC = 25;
soilemp.film = 0.015;
LIDF_file = F(8).FileName;
if ~isempty(LIDF_file)
    canopy.lidf = dlmread(fullfile(path_input, 'leafangles', LIDF_file), '', 3, 0);
end

spectral = define_bands();

leafbio = struct;
leafbio.TDP = define_temp_response_biochem;
soil = struct;

nvars = length(V);
vmax = cellfun(@length, {V.Val})';
vmax(27, 1) = 1;
telmax = max(vmax);
if case_index < 1 || case_index > telmax
    error('case_index must be between 1 and %d', telmax);
end

vi = ones(nvars, 1);
vi(vmax == telmax) = case_index;
xyt = struct;
[xyt.t, xyt.year] = deal(zeros(telmax, 1));

[soil, leafbio, canopy, meteo, angles, xyt] = select_input(V, vi, canopy, options, constants, xyt, soil, leafbio);
canopy.nlayers = ceil(10 * canopy.LAI) + ((meteo.Rin < 200) & options.MoninObukhov) * 60;
canopy.nlayers = max(2, canopy.nlayers);
nl = canopy.nlayers;
x = (-1 / nl:-1 / nl:-1)';
canopy.xl = [0; x];

if isempty(LIDF_file)
    canopy.lidf = leafangles(canopy.LIDFa, canopy.LIDFb);
end

leafbio.emis = 1 - leafbio.rho_thermal - leafbio.tau_thermal;
leafbio.V2Z = 0;

if options.mSCOPE
    error('export_scope_benchmark does not currently support mSCOPE layered leaf inputs');
end

mly = struct;
mly.nly = 1;
mly.pLAI = canopy.LAI;
mly.totLAI = canopy.LAI;
mly.pCab = leafbio.Cab;
mly.pCca = leafbio.Cca;
mly.pCdm = leafbio.Cdm;
mly.pCw = leafbio.Cw;
mly.pCs = leafbio.Cs;
mly.pN = leafbio.N;

atmfile = fullfile(path_input, 'radiationdata', F(4).FileName);
atmo = load_atmo(atmfile, spectral.SCOPEspec);

leafopt = fluspect_mSCOPE(mly, spectral, leafbio, optipar, nl);
leafopt.refl(:, spectral.IwlT) = leafbio.rho_thermal;
leafopt.tran(:, spectral.IwlT) = leafbio.tau_thermal;

if options.soilspectrum == 0
    soil.refl = rsfile(:, soil.spectrum + 1);
else
    soil.refl = BSM(soil, optipar, soilemp);
end
soil.refl(spectral.IwlT) = soil.rs_thermal;

[rad, gap] = RTMo(spectral, atmo, soil, leafopt, canopy, angles, constants, meteo, options); %#ok<ASGLU>
optical_diag = rtmo_observation_diagnostics(spectral, rad, soil, canopy, gap);
global SCOPE_BENCHMARK_TRACE
SCOPE_BENCHMARK_TRACE = struct();
[iter, rad, thermal, soil, bcu, bch, fluxes, resistance, meteo] = ebal(constants, options, rad, gap, meteo, soil, canopy, leafbio, case_index, xyt, integr);
trace = SCOPE_BENCHMARK_TRACE;
clear global SCOPE_BENCHMARK_TRACE

if options.calc_fluor
    fluor_diag = rtmf_source_diagnostics(constants, spectral, rad, soil, leafopt, canopy, gap, angles, bcu.eta, bch.eta);
    rad = RTMf(constants, spectral, rad, soil, leafopt, canopy, gap, angles, bcu.eta, bch.eta);
    Ps = gap.Ps(1:nl);
    Ph = 1 - Ps;
    aPAR_Cab_eta = canopy.LAI * ( ...
        meanleaf(canopy, bch.eta .* rad.Pnh_Cab, 'layers', Ph) + ...
        meanleaf(canopy, bcu.eta .* rad.Pnu_Cab, integr, Ps) ...
    );
    fluor_PoutFrc = leafbio.fqe * aPAR_Cab_eta;
    ep = constants.A * ephoton(spectral.wlF' * 1E-9, constants);
    fluor_EoutFrc = 1E-3 * ep .* (fluor_PoutFrc * optipar.phi(spectral.IwlF));
    sigmaF_raw = pi * rad.LoF_ ./ fluor_EoutFrc;
    fluor_sigmaF = interp1(spectral.wlF(1:4:end), sigmaF_raw(1:4:end), spectral.wlF);
else
    fluor_diag = struct;
    fluor_PoutFrc = nan;
    fluor_EoutFrc = nan(size(spectral.wlF(:)));
    fluor_sigmaF = nan(size(spectral.wlF(:)));
end
rad = RTMt_sb(constants, rad, soil, leafbio, canopy, gap, thermal.Tcu, thermal.Tch, thermal.Tsu, thermal.Tsh, 1, spectral);
if options.calc_planck
    rad = RTMt_planck(spectral, rad, soil, leafopt, canopy, gap, thermal.Tcu, thermal.Tch, thermal.Tsu, thermal.Tsh);
end

IwlP = spectral.IwlP(:);
IwlE = (1:length(spectral.wlE))';
IwlF = spectral.IwlF(:);
IwlT = spectral.IwlT(:);
rtmf_wlE = (400:5:750)';
rtmf_wlF = (640:4:850)';
[~, iwlfi] = intersect(spectral.wlS, rtmf_wlE);
[~, iwlfo] = intersect(spectral.wlS, rtmf_wlF);

benchmark = struct;
benchmark.case_index = case_index;
benchmark.nlayers = nl;
benchmark.integr = integr;
benchmark.source_optipar_file = F(3).FileName;
benchmark.source_atmos_file = F(4).FileName;
benchmark.source_input_file = parameter_files{3};

benchmark.wlP = spectral.wlP(:);
benchmark.wlE = spectral.wlE(:);
benchmark.wlF = spectral.wlF(:);
benchmark.wlT = spectral.wlT(:);
benchmark.rtmf_wlE = rtmf_wlE;
benchmark.rtmf_wlF = rtmf_wlF;

benchmark.leaf_Cab = leafbio.Cab;
benchmark.leaf_Cca = leafbio.Cca;
benchmark.leaf_Cdm = leafbio.Cdm;
benchmark.leaf_Cw = leafbio.Cw;
benchmark.leaf_Cs = leafbio.Cs;
benchmark.leaf_Cant = leafbio.Cant;
benchmark.leaf_Cp = leafbio.Cp;
benchmark.leaf_Cbc = leafbio.Cbc;
benchmark.leaf_N = leafbio.N;
benchmark.leaf_fqe = leafbio.fqe;
benchmark.leaf_rho_thermal = leafbio.rho_thermal;
benchmark.leaf_tau_thermal = leafbio.tau_thermal;

benchmark.biochem_Vcmax25 = leafbio.Vcmax25;
benchmark.biochem_BallBerrySlope = leafbio.BallBerrySlope;
benchmark.biochem_BallBerry0 = leafbio.BallBerry0;
benchmark.biochem_Type = leafbio.Type;
benchmark.biochem_RdPerVcmax25 = leafbio.RdPerVcmax25;
benchmark.biochem_Kn0 = leafbio.Kn0;
benchmark.biochem_Knalpha = leafbio.Knalpha;
benchmark.biochem_Knbeta = leafbio.Knbeta;
benchmark.biochem_stressfactor = leafbio.stressfactor;

benchmark.canopy_LAI = canopy.LAI;
benchmark.canopy_hc = canopy.hc;
benchmark.canopy_zo = canopy.zo;
benchmark.canopy_d = canopy.d;
benchmark.canopy_leafwidth = canopy.leafwidth;
benchmark.canopy_Cd = canopy.Cd;
benchmark.canopy_rwc = canopy.rwc;
benchmark.canopy_kV = canopy.kV;
benchmark.canopy_hot = canopy.hot;
benchmark.canopy_lidf = canopy.lidf(:);

benchmark.soil_spectrum = soil.spectrum;
benchmark.soil_rss = soil.rss;
benchmark.soil_rbs = soil.rbs;
benchmark.soil_rs_thermal = soil.rs_thermal;
benchmark.soil_refl = soil.refl(IwlP);

benchmark.tts = angles.tts;
benchmark.tto = angles.tto;
benchmark.psi = angles.psi;

benchmark.meteo_Ta = meteo.Ta;
benchmark.meteo_ea = meteo.ea;
benchmark.meteo_Ca = meteo.Ca;
benchmark.meteo_Oa = meteo.Oa;
benchmark.meteo_p = meteo.p;
benchmark.meteo_z = meteo.z;
benchmark.meteo_u = meteo.u;
benchmark.meteo_L = meteo.L;

benchmark.Esun_wlP = rad.Esun_(IwlP);
benchmark.Esky_wlP = rad.Esky_(IwlP);
benchmark.Esun_wlE = rad.Esun_(IwlE);
benchmark.Esky_wlE = rad.Esky_(IwlE);
benchmark.Esun_wlT = rad.Esun_(IwlT);
benchmark.Esky_wlT = rad.Esky_(IwlT);
benchmark.Esun_rtmf = rad.Esun_(iwlfi);
benchmark.Esky_rtmf = rad.Esky_(iwlfi);

benchmark.leaf_refl = leafopt.refl(1, IwlP)';
benchmark.leaf_tran = leafopt.tran(1, IwlP)';
benchmark.leaf_Mb = leafopt.Mb(:, :, 1);
benchmark.leaf_Mf = leafopt.Mf(:, :, 1);

	benchmark.canopy_rsd = rad.rsd(IwlP);
	benchmark.canopy_rdd = rad.rdd(IwlP);
	benchmark.canopy_rdo = rad.rdo(IwlP);
	benchmark.canopy_rso = rad.rso(IwlP);
	benchmark.canopy_refl = rad.refl(IwlP);
	benchmark.optical_Emin = rad.Emin_;
	benchmark.optical_Eplu = rad.Eplu_;
	benchmark.optical_Emins = rad.Emins_;
	benchmark.optical_Emind = rad.Emind_;
	benchmark.optical_Eplus = rad.Eplus_;
	benchmark.optical_Eplud = rad.Eplud_;
    benchmark.optical_Po = gap.Po(:);
    benchmark.optical_Pso = gap.Pso(:);
    benchmark.optical_piLocd = optical_diag.piLocd(:);
    benchmark.optical_piLosd = optical_diag.piLosd(:);
    benchmark.optical_piLocu = optical_diag.piLocu(:);
    benchmark.optical_piLosu = optical_diag.piLosu(:);
    benchmark.optical_piLocu_vbvf = optical_diag.piLocu_vbvf(:);
    benchmark.optical_piLocu_w = optical_diag.piLocu_w(:);
    benchmark.optical_piLod = optical_diag.piLod(:);
    benchmark.optical_piLou = optical_diag.piLou(:);

benchmark.fluor_LoF = rad.LoF_(:);
benchmark.fluor_EoutF = rad.EoutF_(:);
benchmark.fluor_PoutFrc = fluor_PoutFrc;
benchmark.fluor_EoutFrc = fluor_EoutFrc(:);
benchmark.fluor_Femleaves = rad.Femleaves_(:);
benchmark.fluor_sigmaF = fluor_sigmaF(:);
benchmark.fluor_LoF_sunlit = rad.LoF_sunlit(:);
benchmark.fluor_LoF_shaded = rad.LoF_shaded(:);
benchmark.fluor_LoF_scattered = rad.LoF_scattered(:);
benchmark.fluor_LoF_soil = rad.LoF_soil(:);
benchmark.fluor_LoutF = rad.LoutF;
benchmark.fluor_EoutF_total = rad.EoutF;
if options.calc_fluor
    benchmark.fluor_MpluEsun = fluor_diag.MpluEsun;
    benchmark.fluor_MminEsun = fluor_diag.MminEsun;
    benchmark.fluor_piLs = fluor_diag.piLs;
    benchmark.fluor_piLd = fluor_diag.piLd;
    benchmark.fluor_Femmin = fluor_diag.Femmin;
    benchmark.fluor_Femplu = fluor_diag.Femplu;
    benchmark.fluor_Fmin = fluor_diag.Fmin;
    benchmark.fluor_Fplu = fluor_diag.Fplu;
end

benchmark.thermal_Lot = rad.Lot_(IwlT);
benchmark.thermal_Eoutte = rad.Eoutte_(IwlT);
benchmark.thermal_Loutt = 0.001 * Sint(rad.Lot_(IwlT), spectral.wlT);
benchmark.thermal_Eoutt = 0.001 * Sint(rad.Eoutte_(IwlT), spectral.wlT);

benchmark.energy_sunlit_eta = bcu.eta(:);
benchmark.energy_shaded_eta = bch.eta(:);
benchmark.energy_sunlit_A = bcu.A(:);
benchmark.energy_shaded_A = bch.A(:);
benchmark.energy_sunlit_Ci = bcu.Ci(:);
benchmark.energy_shaded_Ci = bch.Ci(:);
benchmark.energy_sunlit_rcw = bcu.rcw(:);
benchmark.energy_shaded_rcw = bch.rcw(:);
benchmark.energy_sunlit_gs = (constants.rhoa ./ (constants.Mair * 1E-3)) ./ bcu.rcw(:);
benchmark.energy_shaded_gs = (constants.rhoa ./ (constants.Mair * 1E-3)) ./ bch.rcw(:);
benchmark.energy_Csu = reconstruct_boundary_co2(meteo.Ca, bcu.Ci(:), resistance, canopy.LAI, bcu.rcw(:));
benchmark.energy_Csh = reconstruct_boundary_co2(meteo.Ca, bch.Ci(:), resistance, canopy.LAI, bch.rcw(:));
benchmark.energy_ebu = reconstruct_boundary_vapor(meteo.ea, thermal.Tcu(:), resistance, canopy.LAI, bcu.rcw(:));
benchmark.energy_ebh = reconstruct_boundary_vapor(meteo.ea, thermal.Tch(:), resistance, canopy.LAI, bch.rcw(:));
benchmark.energy_iter_Csu = trace.biochem_input_Csu(:);
benchmark.energy_iter_Csh = trace.biochem_input_Csh(:);
benchmark.energy_iter_ebu = trace.biochem_input_ebu(:);
benchmark.energy_iter_ebh = trace.biochem_input_ebh(:);
benchmark.energy_iter_Tcu = trace.biochem_input_Tcu(:);
benchmark.energy_iter_Tch = trace.biochem_input_Tch(:);
benchmark.energy_iter_Pnu_Cab = trace.biochem_input_Pnu_Cab(:);
benchmark.energy_iter_Pnh_Cab = trace.biochem_input_Pnh_Cab(:);
benchmark.energy_Pnu_Cab = rad.Pnu_Cab(:);
benchmark.energy_Pnh_Cab = rad.Pnh_Cab(:);

benchmark.energy_Tcu = thermal.Tcu(:);
benchmark.energy_Tch = thermal.Tch(:);
benchmark.energy_Tsu = thermal.Tsu;
benchmark.energy_Tsh = thermal.Tsh;

benchmark.energy_Rnuc_sw = rad.Rnuc(:);
benchmark.energy_Rnhc_sw = rad.Rnhc(:);
benchmark.energy_Rnus_sw = rad.Rnus;
benchmark.energy_Rnhs_sw = rad.Rnhs;
benchmark.energy_Rnuct = rad.Rnuct(:);
benchmark.energy_Rnhct = rad.Rnhct(:);
benchmark.energy_Rnust = rad.Rnust;
benchmark.energy_Rnhst = rad.Rnhst;
benchmark.energy_Rnuc = (rad.Rnuc(:) + rad.Rnuct(:));
benchmark.energy_Rnhc = (rad.Rnhc(:) + rad.Rnhct(:));
benchmark.energy_Rnus = rad.Rnus + rad.Rnust;
benchmark.energy_Rnhs = rad.Rnhs + rad.Rnhst;
benchmark.energy_canopyemis = rad.canopyemis;
benchmark.energy_counter = iter.counter;

benchmark.flux_Rnctot = fluxes.Rnctot;
benchmark.flux_lEctot = fluxes.lEctot;
benchmark.flux_Hctot = fluxes.Hctot;
benchmark.flux_Actot = fluxes.Actot;
benchmark.flux_Tcave = fluxes.Tcave;
benchmark.flux_Rnstot = fluxes.Rnstot;
benchmark.flux_lEstot = fluxes.lEstot;
benchmark.flux_Hstot = fluxes.Hstot;
benchmark.flux_Gtot = fluxes.Gtot;
benchmark.flux_Tsave = fluxes.Tsave;
benchmark.flux_Rntot = fluxes.Rntot;
benchmark.flux_lEtot = fluxes.lEtot;
benchmark.flux_Htot = fluxes.Htot;

benchmark.resistance_raa = resistance.raa;
benchmark.resistance_rawc = resistance.rawc;
benchmark.resistance_raws = resistance.raws;
benchmark.resistance_ustar = resistance.ustar;
benchmark.resistance_Kh = resistance.Kh;
benchmark.resistance_uz0 = resistance.uz0;
benchmark.resistance_rai = resistance.rai;
benchmark.resistance_rar = resistance.rar;
benchmark.resistance_rac = resistance.rac;
benchmark.resistance_rws = resistance.rws;
benchmark.resistance_L = trace.resistance_L;

out_dir = fileparts(output_path);
if ~isempty(out_dir) && ~isfolder(out_dir)
    mkdir(out_dir);
end


function diag = rtmf_source_diagnostics(constants, spectral, rad, soil, leafopt, canopy, gap, angles, etau, etah)
wlS = spectral.wlS';
wlF = (640:4:850)';
wlE = (400:5:750)';
[~, iwlfi] = intersect(wlS, wlE);
[~, iwlfo] = intersect(wlS, wlF);
nl = canopy.nlayers;
LAI = canopy.LAI;
litab = canopy.litab;
lazitab = canopy.lazitab;
lidf = canopy.lidf;
nlazi = length(lazitab);
nlinc = length(litab);
nlori = nlinc * nlazi;

Ps = gap.Ps;
Po = gap.Po;
Pso = gap.Pso;
Qs = Ps(1:end-1);

Esunf_ = rad.Esun_(iwlfi);
Eminf_ = rad.Emin_(:, iwlfi)';
Epluf_ = rad.Eplu_(:, iwlfi)';
iLAI = LAI / nl;

Xdd = rad.Xdd(:, iwlfo); %#ok<NASGU>
rho_dd = rad.rho_dd(:, iwlfo);
R_dd = rad.R_dd(:, iwlfo);
tau_dd = rad.tau_dd(:, iwlfo);
vb = rad.vb(:, iwlfo);
vf = rad.vf(:, iwlfo);

Mb = leafopt.Mb;
Mf = leafopt.Mf;

deg2rad = constants.deg2rad;
tto = angles.tto;
tts = angles.tts;
psi = angles.psi;
rs = soil.refl(iwlfo, :);
cos_tto = cos(tto * deg2rad);
sin_tto = sin(tto * deg2rad);
cos_tts = cos(tts * deg2rad);
sin_tts = sin(tts * deg2rad);
cos_ttli = cos(litab * deg2rad);
sin_ttli = sin(litab * deg2rad);
cos_phils = cos(lazitab * deg2rad);
cos_philo = cos((lazitab - psi) * deg2rad);

cds = cos_ttli * cos_tts * ones(1, nlazi) + sin_ttli * sin_tts * cos_phils;
cdo = cos_ttli * cos_tto * ones(1, nlazi) + sin_ttli * sin_tto * cos_philo;
fs = cds / cos_tts;
absfs = abs(fs);
fo = cdo / cos_tto;
absfo = abs(fo);
fsfo = fs .* fo;
absfsfo = abs(fsfo);
foctl = fo .* (cos_ttli * ones(1, nlazi));
fsctl = fs .* (cos_ttli * ones(1, nlazi));
ctl2 = cos_ttli .^ 2 * ones(1, nlazi);

absfs = reshape(absfs, nlori, 1);
absfo = reshape(absfo, nlori, 1);
fsfo = reshape(fsfo, nlori, 1);
absfsfo = reshape(absfsfo, nlori, 1);
foctl = reshape(foctl, nlori, 1);
fsctl = reshape(fsctl, nlori, 1);
ctl2 = reshape(ctl2, nlori, 1);

[MpluEmin, MpluEplu, MminEmin, MminEplu, MpluEsun, MminEsun] = deal(zeros(length(iwlfo), nl));
[U, Fmin, Fplu] = deal(zeros(nl + 1, size(leafopt.Mb, 1)));

Mplu = 0.5 * (Mb + Mf);
Mmin = 0.5 * (Mb - Mf);
for j = 1:nl
    ep = constants.A * ephoton(wlF * 1E-9, constants);
    MpluEmin(:, j) = ep .* (Mplu(:, :, j) * e2phot(wlE * 1E-9, Eminf_(:, j), constants));
    MpluEplu(:, j) = ep .* (Mplu(:, :, j) * e2phot(wlE * 1E-9, Epluf_(:, j), constants));
    MminEmin(:, j) = ep .* (Mmin(:, :, j) * e2phot(wlE * 1E-9, Eminf_(:, j), constants));
    MminEplu(:, j) = ep .* (Mmin(:, :, j) * e2phot(wlE * 1E-9, Epluf_(:, j), constants));
    MpluEsun(:, j) = ep .* (Mplu(:, :, j) * e2phot(wlE * 1E-9, Esunf_, constants));
    MminEsun(:, j) = ep .* (Mmin(:, :, j) * e2phot(wlE * 1E-9, Esunf_, constants));
end

laz = 1 / 36;
if size(etau, 2) < 2
    etau = repmat(etau, 1, 13, 36);
    etau = permute(etau, [2 3 1]);
end

etau_lidf = bsxfun(@times, reshape(etau, nlori, nl), repmat(lidf * laz, 36, 1));
etah_lidf = bsxfun(@times, repmat(etah, 1, nlori)', repmat(lidf * laz, 36, 1));

wfEs = bsxfun(@times, sum(bsxfun(@times, etau_lidf, absfsfo)), MpluEsun) + ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, fsfo)), MminEsun);
vfEplu_h = bsxfun(@times, sum(bsxfun(@times, etah_lidf, absfo)), MpluEplu) - ...
    bsxfun(@times, sum(bsxfun(@times, etah_lidf, foctl)), MminEplu);
vfEplu_u = bsxfun(@times, sum(bsxfun(@times, etau_lidf, absfo)), MpluEplu) - ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, foctl)), MminEplu);
vbEmin_h = bsxfun(@times, sum(bsxfun(@times, etah_lidf, absfo)), MpluEmin) + ...
    bsxfun(@times, sum(bsxfun(@times, etah_lidf, foctl)), MminEmin);
vbEmin_u = bsxfun(@times, sum(bsxfun(@times, etau_lidf, absfo)), MpluEmin) + ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, foctl)), MminEmin);
sfEs = bsxfun(@times, sum(bsxfun(@times, etau_lidf, absfs)), MpluEsun) - ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, fsctl)), MminEsun);
sbEs = bsxfun(@times, sum(bsxfun(@times, etau_lidf, absfs)), MpluEsun) + ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, fsctl)), MminEsun);
sigfEmin_h = bsxfun(@times, sum(etah_lidf), MpluEmin) - ...
    bsxfun(@times, sum(bsxfun(@times, etah_lidf, ctl2)), MminEmin);
sigfEmin_u = bsxfun(@times, sum(etau_lidf), MpluEmin) - ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, ctl2)), MminEmin);
sigbEmin_h = bsxfun(@times, sum(etah_lidf), MpluEmin) + ...
    bsxfun(@times, sum(bsxfun(@times, etah_lidf, ctl2)), MminEmin);
sigbEmin_u = bsxfun(@times, sum(etau_lidf), MpluEmin) + ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, ctl2)), MminEmin);
sigfEplu_h = bsxfun(@times, sum(etah_lidf), MpluEplu) - ...
    bsxfun(@times, sum(bsxfun(@times, etah_lidf, ctl2)), MminEplu);
sigfEplu_u = bsxfun(@times, sum(etau_lidf), MpluEplu) - ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, ctl2)), MminEplu);
sigbEplu_h = bsxfun(@times, sum(etah_lidf), MpluEplu) + ...
    bsxfun(@times, sum(bsxfun(@times, etah_lidf, ctl2)), MminEplu);
sigbEplu_u = bsxfun(@times, sum(etau_lidf), MpluEplu) + ...
    bsxfun(@times, sum(bsxfun(@times, etau_lidf, ctl2)), MminEplu);

piLs = wfEs + vfEplu_u + vbEmin_u;
piLd = vbEmin_h + vfEplu_h;
Fsmin = sfEs + sigfEmin_u + sigbEplu_u;
Fsplu = sbEs + sigbEmin_u + sigfEplu_u;
Fdmin = sigfEmin_h + sigbEplu_h;
Fdplu = sigbEmin_h + sigfEplu_h;
Femmin = iLAI * bsxfun(@times, Qs', Fsmin) + iLAI * bsxfun(@times, (1 - Qs)', Fdmin);
Femplu = iLAI * bsxfun(@times, Qs', Fsplu) + iLAI * bsxfun(@times, (1 - Qs)', Fdplu);

for j = nl:-1:1
    Y(j, :) = (rho_dd(j, :) .* U(j + 1, :) + Femmin(:, j)') ./ (1 - rho_dd(j, :) .* R_dd(j + 1, :));
    U(j, :) = tau_dd(j, :) .* (R_dd(j + 1, :) .* Y(j, :) + U(j + 1, :)) + Femplu(:, j)';
end
for j = 1:nl
    Fmin(j + 1, :) = rad.Xdd(j, iwlfo) .* Fmin(j, :) + Y(j, :);
    Fplu(j, :) = rad.R_dd(j, iwlfo) .* Fmin(j, :) + U(j, :);
end

diag = struct;
diag.leaf_Mb = leafopt.Mb(:, :, 1);
diag.leaf_Mf = leafopt.Mf(:, :, 1);
diag.MpluEsun = MpluEsun;
diag.MminEsun = MminEsun;
diag.piLs = piLs;
diag.piLd = piLd;
diag.Femmin = Femmin;
diag.Femplu = Femplu;
diag.Fmin = Fmin;
diag.Fplu = Fplu;
end
save(output_path, '-struct', 'benchmark');
fprintf('Wrote benchmark fixture to %s\n', output_path);
end


function diag = rtmo_observation_diagnostics(spectral, rad, soil, canopy, gap)
IwlP = spectral.IwlP(:);
nl = canopy.nlayers;
Po = gap.Po;
Po_layers = Po(1:nl);
rs = soil.refl(IwlP, :);

piLosd = rs .* (rad.Emind_(end, IwlP)' * Po(end));
piLod = rad.rdo(IwlP) .* rad.Esky_(IwlP);
piLocd = piLod - piLosd;

piLosu = rs .* (rad.Emins_(end, IwlP)' * Po(end) + rad.Esun_(IwlP) * gap.Pso(end));
piLou = rad.rso(IwlP) .* rad.Esun_(IwlP);
piLocu = piLou - piLosu;
piLocu_vbvf = (sum(rad.vb(:, IwlP) .* Po_layers .* rad.Emins_(1:nl, IwlP) + ...
                  rad.vf(:, IwlP) .* Po_layers .* rad.Eplus_(1:nl, IwlP), 1)') * (canopy.LAI / nl);
piLocu_w = piLocu - piLocu_vbvf;

diag = struct;
diag.piLocd = piLocd;
diag.piLosd = piLosd;
diag.piLocu = piLocu;
diag.piLosu = piLosu;
diag.piLocu_vbvf = piLocu_vbvf;
diag.piLocu_w = piLocu_w;
diag.piLod = piLod;
diag.piLou = piLou;
end


function Cc = reconstruct_boundary_co2(Ca, Ci, resistance, LAI, rcw)
rac = (LAI + 1) * (resistance.raa + resistance.rawc);
Cc = Ca - (Ca - Ci) .* rac ./ (rac + rcw);
end


function ec = reconstruct_boundary_vapor(ea, Tc, resistance, LAI, rcw)
rac = (LAI + 1) * (resistance.raa + resistance.rawc);
ei = satvap_hpa(Tc);
ec = ea + (ei - ea) .* rac ./ (rac + rcw);
end


function es = satvap_hpa(T)
es = 6.107 * 10 .^ (7.5 .* T ./ (237.3 + T));
end


function repo_root = default_repo_root()
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
end


function [parameter_files, options] = load_options(path_input)
fid = fopen('set_parameter_filenames.csv', 'r');
parameter_file = textscan(fid, '%s', 'Delimiter', ',');
fclose(fid);
parameter_files = parameter_file{1};

fid = fopen(fullfile(path_input, parameter_files{1}), 'r');
Ni = textscan(fid, '%d%s', 'Delimiter', ',');
fclose(fid);
N = double(Ni{1});

options = struct;
options.lite = N(1);
options.calc_fluor = N(2);
options.calc_planck = N(3);
options.calc_xanthophyllabs = N(4);
options.soilspectrum = N(5);
options.Fluorescence_model = N(6);
options.apply_T_corr = N(7);
options.verify = N(8);
options.saveCSV = N(9);
options.mSCOPE = N(10);
options.simulation = N(11);
options.calc_directional = N(12);
options.calc_vert_profiles = N(13);
options.soil_heat_method = N(14);
options.calc_rss_rbs = N(15);
options.MoninObukhov = N(16);
options.save_spectral = N(17);
options.Cca_function_of_Cab = 0;
end


function [F, V, options] = load_inputs(path_input, parameter_files, options)
f_names = {'Simulation_Name', 'soil_file', 'optipar_file', 'atmos_file', 'Dataset_dir', ...
    'meteo_ec_csv', 'vegetation_retrieved_csv', 'LIDF_file', 'verification_dir', ...
    'mSCOPE_csv', 'nly'};
cols = {'t', 'year', 'Rin', 'Rli', 'p', 'Ta', 'ea', 'u', 'RH', 'VPD', 'tts', 'tto', 'psi', ...
    'Cab', 'Cca', 'Cdm', 'Cw', 'Cs', 'Cant', 'N', ...
    'SMC', 'BSMBrightness', 'BSMlat', 'BSMlon', ...
    'LAI', 'hc', 'LIDFa', 'LIDFb', ...
    'z', 'Ca', ...
    'Vcmax25', 'BallBerrySlope', 'fqe', ...
    'atmos_names'};
fnc = [f_names, cols];
F = struct('FileID', fnc);

fid = fopen(fullfile(path_input, parameter_files{2}), 'r');
while ~feof(fid)
    line = fgetl(fid);
    if isempty(line)
        continue
    end
    charline = char(line);
    if charline(1) == '%'
        continue
    end
    X = textscan(line, '%s%s', 'Delimiter', ',', 'Whitespace', '\t');
    x = X{1};
    y = X{2};
    k = find(strcmp(fnc, x{:}));
    if ~isempty(k) && ~isempty(y)
        F(k).FileName = y{:};
    end
end
fclose(fid);

fid = fopen(fullfile(path_input, parameter_files{3}), 'r');
k = 1;
clear X
while ~feof(fid)
    line = fgetl(fid);
    y = textscan(line, '%s', 'Delimiter', ',', 'TreatAsEmpty', ' ');
    varnames(k) = y{1}(1); %#ok<AGROW>
    X(k).Val = str2double(y{:}); %#ok<AGROW>
    k = k + 1;
end
fclose(fid);

V = assignvarnames();
for i = 1:length(V)
    j = find(strcmp(varnames, V(i).Name));
    if isempty(j)
        if i == 2
            options.Cca_function_of_Cab = 1;
        else
            error('Required input "%s" not provided in %s', V(i).Name, parameter_files{3});
        end
    else
        idx = find(~isnan(X(j).Val));
        if ~isempty(idx)
            V(i).Val = X(j).Val(idx);
        else
            V(i).Val = -999;
        end
    end
end
end

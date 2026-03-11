"""Top-level package for the PyTorch implementation of SCOPE."""

from importlib.metadata import version as _pkg_version

from .biochem import (
    BiochemicalOptions,
    BiochemicalTemperatureResponse,
    LeafBiochemistryInputs,
    LeafBiochemistryModel,
    LeafBiochemistryResult,
    LeafMeteo,
)
from .config import SimulationConfig
from .canopy.foursail import FourSAILModel, FourSAILResult, campbell_lidf, scope_lazitab, scope_lidf, scope_litab
from .canopy.fluorescence import CanopyBiochemicalFluorescenceResult, CanopyFluorescenceModel, CanopyFluorescenceResult
from .canopy.layered_rt import LayerFluxProfiles, LayeredCanopyTransfer, LayeredCanopyTransportModel
from .canopy.reflectance import CanopyReflectanceModel, CanopyReflectanceResult
from .canopy.thermal import CanopyThermalRadianceModel, CanopyThermalRadianceResult, ThermalOptics, default_thermal_wavelengths
from .spectral.fluspect import FluspectModel, LeafBioBatch, LeafOptics, OptiPar, SpectralGrids
from .spectral.loaders import FluspectResources, SoilSpectraLibrary, load_fluspect_resources, load_scope_filenames, load_soil_spectra
from .spectral.soil import BSMSoilParameters, SoilBSMModel, SoilEmpiricalParams

__all__ = [
    "SimulationConfig",
    "BiochemicalOptions",
    "BiochemicalTemperatureResponse",
    "LeafBiochemistryInputs",
    "LeafBiochemistryModel",
    "LeafBiochemistryResult",
    "LeafMeteo",
    "FluspectModel",
    "LeafBioBatch",
    "LeafOptics",
    "OptiPar",
    "SpectralGrids",
    "FluspectResources",
    "SoilSpectraLibrary",
    "BSMSoilParameters",
    "SoilBSMModel",
    "SoilEmpiricalParams",
    "load_fluspect_resources",
    "load_scope_filenames",
    "load_soil_spectra",
    "FourSAILModel",
    "FourSAILResult",
    "CanopyFluorescenceModel",
    "CanopyFluorescenceResult",
    "CanopyBiochemicalFluorescenceResult",
    "LayerFluxProfiles",
    "LayeredCanopyTransfer",
    "LayeredCanopyTransportModel",
    "CanopyReflectanceModel",
    "CanopyReflectanceResult",
    "CanopyThermalRadianceModel",
    "CanopyThermalRadianceResult",
    "ThermalOptics",
    "campbell_lidf",
    "default_thermal_wavelengths",
    "scope_lazitab",
    "scope_lidf",
    "scope_litab",
]

try:  # pragma: no cover
    __version__ = _pkg_version("scope-torch")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

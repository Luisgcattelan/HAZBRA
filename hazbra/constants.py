"""Constants shared by calculation, post-processing and paper scripts."""

# The 19 CLIMBRA CMIP6 models (order as in the original scripts).
CLIMBRA_MODELS = [
    "ACCESS-CM2", "CMCC-ESM2", "UKESM1-0-LL", "MIROC6", "ACCESS-ESM1-5",
    "MPI-ESM1-2", "HadGEM3-GC31-LL", "KACE", "KIOST", "INM-CM4_8",
    "NESM3", "NorESM2-MM", "INM-CM5", "GFDL-CM4", "IPSL-CM6A-LR",
    "MRI-ESM2", "EC-EARTH3", "GFDL-ESM4", "TaiESM1",
]

# Global warming levels (degC above 1850-1900) plus the 1981-2010 reference ("present").
GWLS = ["present", 1.5, 2, 3]
FUTURE_GWLS = ("1.5", "2", "3")          # string labels used in the NetCDF files

RETURN_PERIODS = [5, 10, 25, 50, 100]
OBS_DATASETS = ["XAVIER", "CHIRPS", "MERGE"]

# Indicator -> hazard category (= first folder level of the product).
INDICATOR_HAZARD = {
    "Rx1day": "Rainfall",
    "Rx5day": "Rainfall",
    "CWD": "Rainfall",
    "prcptot": "Rainfall",
    "pr_season": "Rainfall",
    "CDD": "Drought",
    "CDD_season": "Drought",
    "SPI3": "Drought",
    "SPI6": "Drought",
    "TXX": "Heat",
    "TNN": "Heat",
    "Frost": "Heat",
    "WSDI": "Heat",
    "FWI": "Compound",
    "HI": "Compound",
    "Dry_Hot": "Compound",
}

# Name of the main data variable inside each indicator's files (multi-variable indicators
# store several variables; this is the one used for maps/validation).
INDICATOR_MAIN_VAR = {k: k for k in INDICATOR_HAZARD}
INDICATOR_MAIN_VAR.update({"FWI": "FWI_max", "HI": "HI_max"})

INDICATOR_UNITS = {
    "Rx1day": "mm/day", "Rx5day": "mm/5days", "CWD": "days", "prcptot": "mm/year",
    "pr_season": "mm/month", "CDD": "days", "CDD_season": "days", "SPI3": "-", "SPI6": "-",
    "TXX": "°C", "TNN": "°C", "Frost": "days", "WSDI": "days",
    "FWI": "FWI", "HI": "°C", "Dry_Hot": "days",
}

# Indicators without GEV products (no *_return_period / *_params files).
NO_RETURN_PERIOD = {"Dry_Hot", "prcptot", "pr_season", "Frost", "WSDI"}

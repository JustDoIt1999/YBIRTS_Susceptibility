from __future__ import annotations

import ee

SMDP_ASSET_VERSION_SUFFIX = "_v2"

STATIC_FACTOR_SPECS = [
    {"factor_name": "Slope", "band_name": "Slope"},
    {"factor_name": "Aspect", "band_name": "Aspect"},
    {"factor_name": "TWI", "band_name": "TWI"},
    {"factor_name": "Local_Relief", "band_name": "Local_Relief", "asset_name": "Local_Relief_5km"},
    {"factor_name": "Soil", "band_name": "Soil"},
    {"factor_name": "GIC", "band_name": "GIC"},
    {"factor_name": "Vegetation_type", "band_name": "Vegetation_type"},
]

DYNAMIC_ANNUAL_FACTOR_SPECS = [
    {"factor_name": "TDD", "band_name": "TDD"},
    {"factor_name": "SMDP", "band_name": "SMDP"},
    {"factor_name": "Precipitation", "band_name": "Precipitation"},
    {"factor_name": "WSTmax", "band_name": "WSTmax"},
    {"factor_name": "WSRH", "band_name": "WSRH"},
    {"factor_name": "NDVI", "band_name": "NDVI"},
    {"factor_name": "Summer_SoilMoisture", "band_name": "Summer_SoilMoisture"},
]

RANK_STATIC_FEATURE_COLUMNS = [item["band_name"] for item in STATIC_FACTOR_SPECS]
RANK_DYNAMIC_FEATURE_COLUMNS = [item["band_name"] for item in DYNAMIC_ANNUAL_FACTOR_SPECS]
RANK_FEATURE_OUTPUT_COLUMNS = [*RANK_STATIC_FEATURE_COLUMNS, *RANK_DYNAMIC_FEATURE_COLUMNS]

ASSET_DTYPE_BY_BAND = {
    "Slope": "float32",
    "Aspect": "int16",
    "TWI": "float32",
    "Local_Relief": "float32",
    "Soil": "float32",
    "Vegetation_type": "int16",
    "GIC": "float32",
    "TDD": "float32",
    "SMDP": "float32",
    "Precipitation": "float32",
    "WSTmax": "float32",
    "WSRH": "float32",
    "NDVI": "float32",
    "Summer_SoilMoisture": "float32",
}


def expected_asset_dtype(band_name: str) -> str:
    normalized = str(band_name).strip()
    if normalized not in ASSET_DTYPE_BY_BAND:
        raise KeyError(f"Undefined band dtype policy: {band_name}")
    return ASSET_DTYPE_BY_BAND[normalized]


def cast_single_band_for_asset(image: ee.Image, band_name: str) -> ee.Image:
    normalized = str(band_name).strip()
    band_image = ee.Image(image).select([normalized])
    dtype = expected_asset_dtype(normalized)
    if dtype == "int16":
        return band_image.toInt16()
    if dtype == "float32":
        return band_image.toFloat()
    raise ValueError(f"Unsupported asset dtype: {dtype}; band={normalized}")


def cast_stack_for_asset(image: ee.Image, band_names: list[str]) -> ee.Image:
    stack = None
    for band_name in band_names:
        typed_band = cast_single_band_for_asset(image, band_name)
        stack = typed_band if stack is None else stack.addBands(typed_band)
    return ee.Image(stack).select(band_names)


def cast_stack_for_tabular_sampling(image: ee.Image, band_names: list[str], *, nodata_value: float) -> ee.Image:
    stack = None
    for band_name in band_names:
        typed_band = ee.Image(image).select([band_name]).toFloat().unmask(float(nodata_value), False)
        stack = typed_band if stack is None else stack.addBands(typed_band)
    return ee.Image(stack).select(band_names)


def factor_library_root(runtime_root: str) -> str:
    return f"{runtime_root.rstrip('/')}/environmental_factors/factor_library_v2"


def factor_group_root(runtime_root: str, group: str) -> str:
    normalized_group = str(group).strip().lower()
    if normalized_group == "static":
        return f"{factor_library_root(runtime_root)}/static"
    if normalized_group == "dynamic_annual":
        return f"{factor_library_root(runtime_root)}/dynamic_annual"
    raise ValueError(f"Unsupported factor asset group: {group}")


def factor_asset_name(group: str, factor_name: str, year: int | None = None) -> str:
    normalized_group = str(group).strip().lower()
    normalized_factor = str(factor_name).strip()
    if normalized_group == "static":
        factor_spec = next(
            (item for item in STATIC_FACTOR_SPECS if item["factor_name"] == normalized_factor),
            None,
        )
        asset_name = str((factor_spec or {}).get("asset_name") or normalized_factor)
        return f"{asset_name}_v2"
    if normalized_group == "dynamic_annual":
        if year is None:
            raise ValueError("dynamic_annual asset name requires year.")
        if normalized_factor == "SMDP":
            return f"{normalized_factor}_{int(year)}{SMDP_ASSET_VERSION_SUFFIX}"
        return f"{normalized_factor}_{int(year)}_v2"
    raise ValueError(f"Unsupported factor asset group: {group}")


def factor_asset_id(runtime_root: str, group: str, factor_name: str, year: int | None = None) -> str:
    return f"{factor_group_root(runtime_root, group).rstrip('/')}/{factor_asset_name(group, factor_name, year=year)}"

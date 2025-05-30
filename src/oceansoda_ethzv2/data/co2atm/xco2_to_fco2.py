import pyseaflux as sf
import xarray as xr


def xco2_to_fco2(
    xco2: xr.DataArray,
    temperature_degC: xr.DataArray,
    pressure_Pa: xr.DataArray,
    salinity: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Convert xCO2 to fCO2 using the CO2SYS algorithm.

    Parameters:
        xco2 (xr.DataArray): The xCO2 data.
        temperature_degC (xr.DataArray): The temperature data (in degrees Celsius).
        pressure_Pa (xr.DataArray): The pressure data (in Pascals).
        salinity (xr.DataArray, optional): The salinity data. Defaults to None.

    Returns:
        xr.DataArray: The fCO2 data.
    """

    temp_degK = temperature_degC + 273.15  # Convert temperature from Celsius to Kelvin
    press_hpa = pressure_Pa / 100.0  # Convert pressure from Pa to hPa
    press_atm = press_hpa / 1013.25  # Convert pressure from hPa to atm

    pH2O = sf.vapour_pressure.dickson2007(salinity, temp_degK)
    press_corrected_atm = press_atm - pH2O

    # Convert xCO2 to pCO2
    pCO2 = xco2 * press_corrected_atm
    # Convert pCO2 to fCO2
    fCO2 = sf.pCO2_to_fCO2(
        pCO2,
        tempSW_C=temperature_degC,
        pres_hPa=press_hpa,
    )

    fCO2.attrs["units"] = "µatm"
    fCO2.attrs["description"] = (
        "fCO2 calculated from xCO2 using the pySeaFlux. "
        "1. Correct pressure for water vapour using Dickson 2007 method. "
        "2. Convert xCO2 to pCO2 using corrected pressure. "
        "3. Convert pCO2 to fCO2 with pCO2 * virial expansion factor."
    )

    return fCO2.rename("fco2")

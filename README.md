## Large-scale Agriculture Water System model (LAWS)

LAWS is an open-source coupled water and crop model built by extending the
[Community Water Model (CWatM)](https://github.com/iiasa/CWatM). It retains
CWatM's hydrological core and adds daily crop growth, crop-specific
evapotranspiration, and root-zone water-uptake processes.


## LAWS agricultural modules

The original LAWS crop implementation is kept in three source modules under
`laws/agricultural_modules/`:

- `BioMassCalc.py`: crop growth, biomass, phenology, and yield calculations.
- `new_PM_ET.py`: crop-specific Penman-Monteith evapotranspiration.
- `crop_WU.py`: root-zone crop water uptake and water-stress calculations.

Enable the coupled crop model in a settings file with:

```ini
[OPTIONS]
crop_coupling = True
includeCrops = True
```

Both options must be set to `True` to run the coupled LAWS crop processes.

## Running LAWS

Run the model from the repository root with:

```bash
python run_laws.py settings.ini
```

## Provenance, license, and citation

LAWS is a modified version of CWatM and is distributed under the GNU General
Public License v3.0 or later. The original CWatM copyright and license notices
must be retained. LAWS-specific crop-module code and modifications are identified
separately in the source and project notice.

For CWatM documentation and scientific background, see the
[official CWatM documentation](https://cwatm.iiasa.ac.at/). Scientific work that
uses LAWS should cite both the eventual LAWS release/citation and the applicable
CWatM publications.

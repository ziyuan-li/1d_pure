# Raw benchmark datasets

This directory contains the six public datasets added by the JoCS extension
plan plus a local copy of California Housing. Each UCI dataset directory keeps
the original archive as `source.zip`; California Housing keeps the archive used
by scikit-learn as `source.tgz`.

| Dataset | UCI ID | Rows | Input features | Regression target(s) | Main extracted file |
|---|---:|---:|---:|---|---|
| Energy Efficiency | 242 | 768 | 8 | `Y1` heating load, `Y2` cooling load | `energy_efficiency/ENB2012_data.xlsx` |
| Airfoil Self-Noise | 291 | 1,503 | 5 | scaled sound pressure level | `airfoil_self_noise/airfoil_self_noise.dat` |
| Yacht Hydrodynamics | 243 | 308 | 6 | residuary resistance | `yacht_hydrodynamics/yacht_hydrodynamics.data` |
| Combined Cycle Power Plant | 294 | 9,568 | 4 | `PE` net electrical energy output | `combined_cycle_power_plant/CCPP/Folds5x2_pp.xlsx` |
| Superconductivity Tc | 464 | 21,263 | 81 | `critical_temp` | `superconductivity/train.csv` |
| Bike Sharing (hourly) | 275 | 17,379 | 12 usable before encoding | `cnt` total rentals | `bike_sharing/hour.csv` |
| California Housing | - | 20,640 | 8 | median house value | `california_housing/CaliforniaHousing/cal_housing.data` |

The six UCI modeling tables contain no missing values in their original
release. The scikit-learn documentation also reports no missing attributes for
California Housing.

## Official sources

- Energy Efficiency: <https://archive.ics.uci.edu/dataset/242/energy+efficiency>
- Airfoil Self-Noise: <https://archive.ics.uci.edu/dataset/291/airfoil+self+noise>
- Yacht Hydrodynamics: <https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics>
- Combined Cycle Power Plant: <https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant>
- Superconductivty Data: <https://archive.ics.uci.edu/dataset/464/superconductivty+data>
- Bike Sharing: <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>
- California Housing: <https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset>

The UCI pages list these releases under the Creative Commons Attribution 4.0
International license. Cite the individual dataset DOI shown on its UCI page
when publishing results.

## Notes

- Energy Efficiency has two valid regression targets. The experiment protocol
  must state whether `Y1`, `Y2`, or both are evaluated.
- The Combined Cycle Power Plant archive also contains an ODS copy; use the
  XLSX file for the Python pipeline.
- The superconductivity archive contains `unique_m.csv` with chemical formula
  information. The 81-feature benchmark described in the plan uses `train.csv`.
- Bike Sharing uses the hourly table. When predicting `cnt`, exclude `casual`
  and `registered` because their sum is exactly the target; exclude `instant`
  and use `dteday` only to construct the chronological split.
  Predict concentration from 128 sensor responses, with gas identity and drift
  batch as context. Preserve batch order: train on 1-6, validate on 7-8, and
  test on 9-10.

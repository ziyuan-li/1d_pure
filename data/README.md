# Data

`synthetic/` contains the three generated datasets used in the paper, with
arrays `X`, `y`, and the noise-free `signal`. Regenerate them with:

```bash
python -m pure_resmlp.paper generate-data
```

The pairwise and higher-order tasks use generation seed 0, 20 irrelevant
features, and Gaussian response noise with SD 5% of the clean signal SD.
Friedman uses `make_friedman1`, generation seed 0, and no response noise.
All three contain 10,000 rows. Exact split indices and fitted scalers are in
`protocol/`; the synthetic demo never needs external data.

Download real datasets separately to the following paths under `data/`.

| Dataset source | Local file |
|---|---|
| [Concrete](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | `concrete+compressive+strength/Concrete_Data.xls` |
| [California Housing](https://ndownloader.figshare.com/files/5976036) | `raw/california_housing/CaliforniaHousing/cal_housing.data` |
| [Energy](https://archive.ics.uci.edu/dataset/242/energy+efficiency) | `raw/energy_efficiency/ENB2012_data.xlsx` |
| [Yacht](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics) | `raw/yacht_hydrodynamics/yacht_hydrodynamics.data` |
| [Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg) | `raw/auto_mpg/auto-mpg.data` |
| [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | `raw/bike_sharing/hour.csv` |
| [Parkinsons](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) | `raw/parkinsons_telemonitoring/parkinsons_updrs.data` |

Alternatively, set `PURE_DATA_DIR` to a separate directory with this layout.
The loaders read local files and never download data automatically. They use
the manuscript's encoding, training-only scaling, and chronological group
boundaries. Evaluation checks the saved indices and scaling statistics before
using a checkpoint. For direct inference, `predict_encoded` in
`pure_resmlp.paper` accepts the full encoded feature order documented in the
run's preprocessing JSON, before scaling or context-column removal.

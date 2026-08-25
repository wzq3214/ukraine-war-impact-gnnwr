# Ukraine war impact GNNWR reproducibility

This repository contains the computational workflow supporting the spatial regression analysis reported in Section 3.4 of the associated manuscript. It reproduces the five-fold spatial-block validation, GNNWR, GWR and OLS comparison, final GNNWR fit, local coefficient estimates and Figure 6.

The workflow uses the public `gnnwr` package and does not modify or reimplement its model core. Tiny-GNNWR and other surrogate implementations are not used.

## Analytical specification

| Role | Workbook field |
|---|---|
| Response | `Disp_Rate_FINNAL` |
| CII | `CII_pos` |
| Older adult share | `Index_Agin` |
| Infrastructure damage density | `ALL_Damage_Density` |
| Nighttime light extinction ratio | `Extinguished_Ratio` |
| Projected coordinates | `proj_X`, `proj_Y` |
| ADM3 identifier | `ADM3_PCODE` |

The response is treated as a proportion and transformed once with the standard logit function. Records are retained when the response is positive and the projected coordinates are valid. No fixed sample count is imposed.

`CII_pos` is the computational CII field. When `CII` is also present, the workflow tests whether `CII_pos` differs from it only by an additive constant and records the result in `Data_audit.csv`.

## Methods reproduced

- five-fold spatial-block cross-validation
- inner spatial validation for the GNNWR epoch and GWR neighbour count
- training-fold-only imputation and standardisation
- identical outer test folds for GNNWR, GWR and OLS
- pooled out-of-fold R², RMSE and MAE on the logit scale
- residual Global Moran's I with a permutation p value
- final full-sample GNNWR fit using the median selected epoch
- standardized and raw-scale local coefficient exports
- Figure 6 with coefficients shown only for modelled ADM3 units

No zero filling or spatial interpolation is applied to unestimated local coefficients. The coefficient maps describe conditional spatial associations. They are not causal effects, significance maps or local mechanisms.

## Software

The analysis is pinned to `gnnwr==0.1.17`. The workflow records the installed package version, source location, Python environment, input hashes and run configuration in `Run_manifest.json`.

Primary references

- Du, Z., Wang, Z., Wu, S., Zhang, F., and Liu, R. 2020. *Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity*. International Journal of Geographical Information Science, 34, 1353–1377. DOI `10.1080/13658816.2019.1707834`.
- Yin, Z., Ding, J., Liu, Y., Wang, R., Wang, Y., Chen, Y., Qi, J., Wu, S., and Du, Z. 2024. *GNNWR: an open-source package of spatiotemporal intelligent regression methods for modeling spatial and temporal nonstationarity*. Geoscientific Model Development, 17, 8455–8468. DOI `10.5194/gmd-17-8455-2024`.

## Repository structure

```text
ukraine-war-impact-gnnwr/
├── reproducibility/
│   └── section_3_4_gnnwr_official_workflow.py
├── data/
│   └── README.md
├── tests/
│   └── test_workflow_utilities.py
├── .github/workflows/tests.yml
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── NOTICE
└── README.md
```

Research data and administrative boundary files are not distributed in this repository. See `data/README.md` for required file names and fields.

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate ukraine-war-impact-gnnwr
```

### Pip

Create a Python 3.10 environment and run

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pre-run checks

```bash
python reproducibility/section_3_4_gnnwr_official_workflow.py --self-test

python reproducibility/section_3_4_gnnwr_official_workflow.py \
  --data data/Section3_4_GNNWR_input.xlsx \
  --output outputs/section_3_4_gnnwr \
  --audit-only
```

Inspect `Data_audit.csv`, `Predictor_correlation.csv` and `Predictor_VIF.csv` before formal modelling.

## Formal run

```bash
python -u reproducibility/section_3_4_gnnwr_official_workflow.py \
  --data data/Section3_4_GNNWR_input.xlsx \
  --boundary data/UKR_ADM3_2023.shp \
  --boundary-id-col ADM3_PCODE \
  --output outputs/section_3_4_gnnwr \
  --device auto
```

For an algorithm-stability diagnostic, add

```bash
--final-seeds 42,43,44,45,46
```

The manuscript map should retain the prespecified primary seed unless the manuscript explicitly states that median coefficients across seeds were mapped.

## Figure 6a

The analytical workbook contains the IDP stock ratio. Exact reproduction of the count classes used in the current Figure 6a requires a separate table with raw stock counts.

```bash
python -u reproducibility/section_3_4_gnnwr_official_workflow.py \
  --data data/Section3_4_GNNWR_input.xlsx \
  --boundary data/UKR_ADM3_2023.shp \
  --boundary-id-col ADM3_PCODE \
  --stock-table data/IDP_stock_2024_ADM3.xlsx \
  --stock-table-id-col ADM3_PCODE \
  --stock-col IDP_stock_2024 \
  --output outputs/section_3_4_gnnwr
```

Without the stock table, panel a is explicitly labelled and plotted as an IDP stock-ratio map. The workflow never reconstructs stock counts from the ratio without a verified denominator.

## Main outputs

- `Table3_model_performance.csv`
- `Table3_model_performance_rounded.csv`
- `All_models_spatial_OOF_predictions.csv`
- `Model_metrics_by_fold.csv`
- `Spatial_fold_assignments.csv`
- `GWR_inner_validation_bandwidth_search.csv`
- `GNNWR_local_coefficients_primary.csv`
- `GNNWR_local_coefficients_all_seeds.csv`
- `GNNWR_local_coefficient_seed_stability.csv`
- `Figure6_section3_4.png`
- `Figure6_section3_4.pdf`
- `Figure6_section3_4.tif`
- `Run_manifest.json`

Table 3 values must be copied from the formal output files and are never hard-coded in the script.

## Citation

Use the repository citation shown in `CITATION.cff`, together with the two GNNWR method references above.

## Licence

This repository is distributed under the GNU General Public License v3.0. The official GNNWR package is a separate upstream dependency distributed under the same licence.

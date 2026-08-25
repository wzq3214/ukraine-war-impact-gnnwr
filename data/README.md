# Data requirements

Data are not distributed in this repository. Place the following files in this directory before running the workflow.

## Required analytical workbook

File name

`Section3_4_GNNWR_input.xlsx`

Required fields

- `ADM3_PCODE`
- `Disp_Rate_FINNAL`
- `CII_pos`
- `Index_Agin`
- `ALL_Damage_Density`
- `Extinguished_Ratio`
- `proj_X`
- `proj_Y`

Optional output coordinates

- `POINT_X`
- `POINT_Y`

The workflow retains positive, finite response values with valid projected coordinates. It does not enforce a fixed number of observations.

## Required ADM3 boundary

Use a polygon boundary named `UKR_ADM3_2023.shp` together with its `.dbf`, `.shx`, `.prj` and, where available, `.cpg` sidecar files. The join field must correspond to `ADM3_PCODE` unless another field is supplied with `--boundary-id-col`.

## Optional stock-count table

To reproduce the count classes in Figure 6a, provide `IDP_stock_2024_ADM3.xlsx` with

- `ADM3_PCODE`
- `IDP_stock_2024`

Do not infer stock counts from the ratio unless the population denominator has been independently verified.

## Data governance

Before redistribution, verify the licences and terms of the IDMC, administrative boundary and derived analytical datasets. Do not commit confidential, restricted or personally identifiable data.

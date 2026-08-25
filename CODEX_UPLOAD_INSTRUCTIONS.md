# Codex upload instruction

Use the attached release-candidate directory as the source of truth and publish it to `wzq3214/ukraine-war-impact-gnnwr`.

1. Confirm the authenticated account is `wzq3214` and the repository is public.
2. Clone `https://github.com/wzq3214/ukraine-war-impact-gnnwr` and work on branch `main`.
3. Preserve the existing `LICENSE` file.
4. Replace the existing `README.md` with the supplied `README.md`.
5. Add all supplied files and directories except `.gitignore.append`.
6. Append the contents of `.gitignore.append` to the existing `.gitignore` once. Do not duplicate entries.
7. Do not upload analytical data, ADM3 shapefiles, model checkpoints, TensorBoard logs, generated figures or result files.
8. Run the following checks from the repository root.

```bash
python -m py_compile reproducibility/section_3_4_gnnwr_official_workflow.py
python reproducibility/section_3_4_gnnwr_official_workflow.py --self-test
pytest -q
```

9. Confirm that the workflow imports the official `gnnwr==0.1.17` package, uses `Disp_Rate_FINNAL` as the response, uses `CII_pos`, `Index_Agin`, `ALL_Damage_Density` and `Extinguished_Ratio` as the four covariates, and contains no Tiny-GNNWR, zero filling, spatial interpolation or dominant-mechanism output.
10. Commit directly to `main` with the message

`Add validated Section 3.4 GNNWR reproducibility workflow`

11. Push the commit and report the commit SHA, the repository link, the main script link and the GitHub Actions run link.

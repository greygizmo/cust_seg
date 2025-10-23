# CSV Enrichment Checklist

## 0) Guardrails

- [ ] Keep output file name exactly `icp_scored_accounts.csv` (downstream contract).
- [ ] Do not introduce Big-Box/Small-Box fields anywhere—taxonomy is SuperDivision / Division / SubDivision.
- [ ] No ML libraries; only pandas/numpy.

## 1) Create modules

- [ ] Add folder `/features/` with files:
  - [ ] `__init__.py`
  - [ ] `product_taxonomy.py` (join & validation)
  - [ ] `spend_dynamics.py`
  - [ ] `adoption_and_mix.py`
  - [ ] `health_concentration.py`
  - [ ] `sw_hw_whitespace.py`
  - [ ] `pov_tags.py`

## 2) Update config

- [ ] Append the `[data_sources]`, `[momentum_weights]`, `[windows]` sections to `config.toml` as shown.

## 3) Wire into goe_icp_scoring.py

- [ ] Import the new modules.
- [ ] Load sales + products from `config.toml`.
- [ ] Determine `as_of_date` (max txn date if unset).
- [ ] Call:
  - [ ] `validate_and_join_products(...)`
  - [ ] `compute_spend_dynamics(...)`
  - [ ] `compute_adoption_and_mix(...)`
  - [ ] `month_hhi_12m(...)`, `discount_pct(...)`
  - [ ] `sw_dominance_and_whitespace(...)`
  - [ ] Merge features → compute `momentum_score` using config weights.
  - [ ] `make_pov_tags(...)` → merge tags.
- [ ] Merge features with existing `df_accounts` (ICP) on `account_id`.
- [ ] Add `as_of_date`, `run_timestamp_utc`.
- [ ] Write `icp_scored_accounts.csv` once, after merge.

## 4) Column QA (spot checks)

- [ ] Verify the presence of all new columns in the CSV (see schema).
- [ ] For 2–3 hand-picked accounts:
  - [ ] Check `spend_13w`, `spend_13w_prior`, `delta_13w_pct` by manual filter in pandas.
  - [ ] Check `slope_13w` by plotting 13 weekly points and confirming sign & magnitude.
  - [ ] Check `top_subdivision_12m` equals the LTM max bucket label.
  - [ ] Check `month_conc_hhi_12m` rises when revenue is lumpy.
  - [ ] Check `pov_primary` precedence holds (if multiple flags true, only the first in order appears).

## 5) Performance sanity

- [ ] Ensure groupby key cardinalities are indexed: `account_id`, `date`.
- [ ] If runtime > acceptable: pre-filter to LTM for all features except `median_interpurchase_days` (which uses history).
- [ ] Consider saving intermediate Parquet tables (`transactions_joined.parquet`) for faster dev cycles (not required for prod).

## 6) Documentation

- [ ] In README, note: “All List-Builder features are computed in-repo and emitted in `icp_scored_accounts.csv` for Power BI consumption. No DAX measures.”
- [ ] Document any threshold changes in `pov_tags.py` top constants.

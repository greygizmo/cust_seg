# Power BI Dashboard Guide for ICP Scoring

This guide describes a practical, opinionated Power BI dashboard built on top of the outputs already produced by this repo:

- `data/processed/icp_scored_accounts.csv` – core scored accounts dataset.
- `artifacts/account_neighbors.csv` – similar accounts for “look-alike” targeting.
- `reports/call_lists/YYYYMMDD/*.csv` – curated call-list exports (optional, if you adopt the CLI).

It assumes **no new data sources** beyond what the pipeline already generates.

---

## 1. Recommended Data Model

### 1.1 Tables

- **`icp_scored_accounts` (Fact)**  
  - Source: `data/processed/icp_scored_accounts.csv`  
  - Grain: one row per customer.  
  - Key columns:
    - `Customer ID` (text; canonical ID – use as the model key).
    - `Company Name`
    - `ICP_score_hardware`, `ICP_grade_hardware`
    - `ICP_score_cre`, `ICP_grade_cre`
    - `Hardware_score`, `Software_score`
    - `GP_Since_2023_Total`, `GP_T4Q_Total`, `GP_LastQ_Total`, `GP_PrevQ_Total`, `GP_QoQ_Growth`
    - `Industry`, `Industry Sub List`, `Industry_Reasoning`
    - `am_sales_rep`, `AM_Territory`, `activity_segment`
    - `adoption_assets`, `adoption_profit`, `cre_adoption_assets`, `cre_adoption_profit`
    - `relationship_profit`, `cre_relationship_profit`
    - `printer_count` (from features), plus printer/rollup columns.

- **`account_neighbors` (Optional fact-like helper)**  
  - Source: `artifacts/account_neighbors.csv`  
  - Grain: one row per (`account_id`, `neighbor_account_id`).  
  - Use as a detail table to support “similar accounts” pop-outs; relate `account_id` to `Customer ID` in `icp_scored_accounts` (both as text).

- **`call_lists` (Optional fact)**  
  - Source: `reports/call_lists/YYYYMMDD/*.csv` (when you run the export CLI).  
  - Grain: one row per call-list row (ranked account).  
  - Join on `Customer ID` back into `icp_scored_accounts` for additional context and performance tracking.

### 1.2 Simple Dimensions (Optional)

You can create three small dimension tables directly in Power BI by referencing the main fact:

- **`DimIndustry`**
  - Columns: `Industry`, `Industry Sub List`.
  - Relationship: `DimIndustry[Industry] 1-* icp_scored_accounts[Industry]`.
  - Use for slicers and to avoid duplicated industry labels on visuals.

- **`DimTerritory`**
  - Columns: `AM_Territory`, optionally `am_sales_rep` for hierarchies.
  - Relationship: `DimTerritory[AM_Territory] 1-* icp_scored_accounts[AM_Territory]`.

- **`DimGrade`**
  - Columns: `ICP_grade_hardware`, `ICP_grade_cre` (or a unified `grade` if you choose one division).
  - Use as the axis for grade-mix visuals.

These are convenience tables: you can also drive everything directly off the single wide fact if you prefer.

---

## 2. Loading Data into Power BI

### 2.1 Load from CSV (recommended to start)

1. In Power BI Desktop: **Home → Get Data → Text/CSV**.
2. Browse to `data/processed/icp_scored_accounts.csv`.
3. In the preview, ensure **`Customer ID` is type *Text*** (not Whole Number or Decimal).
4. Load. Rename the table to `icp_scored_accounts` (or `icp_scored_accounts v1` if you want to match the existing measures file).
5. Repeat for:
   - `artifacts/account_neighbors.csv` (optional)
   - One of the dated folders under `reports/call_lists/` (optional)

### 2.2 Load from SQL (optional)

If you publish to `dbo.customer_icp` via `publish_scored_to_db.py`:

1. **Get Data → SQL Server**.
2. Server: use `AZSQL_SERVER`; Database: `ICP_AZSQL_DB` (from `.env`).
3. Select `dbo.customer_icp` and import.
4. Ensure `Customer ID` is text and otherwise follow the same modeling guidance.

---

## 3. Core Measures (DAX)

Much of this already exists in `docs/powerbi tmdl/powerbi_measures.txt`. Below is a simplified, human-readable set of measures that align with the current schema.

Assume the main table is called `'icp_scored_accounts v1.5'` with `Customer ID` as the key.

### 3.1 Base Measures

```DAX
Customers =
    DISTINCTCOUNT ( 'icp_scored_accounts v1.5'[Customer ID] )

Profit :=
    SUM ( 'icp_scored_accounts v1.5'[GP_Since_2023_Total] )

Customers_HW =
    CALCULATE (
        [Customers],
        'icp_scored_accounts v1.5'[Hardware_score] > 0
    )

Customers_SW =
    CALCULATE (
        [Customers],
        'icp_scored_accounts v1.5'[Software_score] > 0
    )
```

### 3.2 Percent-of-Total Metrics

```DAX
Pct Customers =
    DIVIDE (
        [Customers],
        CALCULATE ( [Customers], ALL ( 'icp_scored_accounts v1.5' ) )
    )

Pct Profit =
    DIVIDE (
        [Profit],
        CALCULATE ( [Profit], ALL ( 'icp_scored_accounts v1.5' ) )
    )
```

You can apply the same pattern to `Customers_HW`, `Customers_SW`, or any other segment.

### 3.3 A/B Coverage Metrics (Hardware)

```DAX
Customers_AB_HW =
    CALCULATE (
        [Customers],
        'icp_scored_accounts v1.5'[ICP_grade_hardware] IN { "A", "B" }
    )

Pct_Customers_AB_HW =
    DIVIDE (
        [Customers_AB_HW],
        CALCULATE ( [Customers], ALL ( 'icp_scored_accounts v1.5' ) )
    )

Profit_AB_HW =
    CALCULATE (
        [Profit],
        'icp_scored_accounts v1.5'[ICP_grade_hardware] IN { "A", "B" }
    )

Pct_Profit_AB_HW =
    DIVIDE (
        [Profit_AB_HW],
        CALCULATE ( [Profit], ALL ( 'icp_scored_accounts v1.5' ) )
    )
```

These measures drive score/grade coverage visuals for leadership and territory managers.

---

## 4. Applying the TMDL Measures

To avoid hand-maintaining DAX, this repo includes a TMDL script you can apply directly:

- File: `docs/powerbi tmdl/powerbi_measures_clean.tmdl`
- It defines:
  - Core portfolio measures (`Customers`, `Profit`, percent-of-total).
  - Hardware A/B coverage measures (suffix `_HW`).
  - CRE A/B coverage measures (suffix `_CRE`).
  - HW vs SW revenue share.
  - Neighbor helper measures on `account_neighbors`.

**How to apply in Power BI Desktop:**

1. Open your `.pbix` and switch to **Model** view.
2. On the ribbon, choose **External Tools → TMDL View** (or "View TMDL" in newer builds).
3. In the TMDL script editor:
   - Paste the contents of `powerbi_measures_clean.tmdl`.
   - Ensure the table names (`'icp_scored_accounts v1.5'` and `'account_neighbors'`) match your model.
4. Click **Apply**. Power BI will create/update the measures on those tables.

Once applied, you can drag these measures onto visuals as described below.

---

## 5. Suggested Pages & Visuals

### 5.1 Executive Overview (Portfolio, HW + CRE)

**Purpose:** Show total portfolio quality and A/B coverage for both divisions.

**Recommended layout:**

- **Top row – KPIs (cards):**
  - Portfolio:
    - `Customers`
    - `Profit`
    - `'Pct Customers'`
    - `'Pct Profit'`
  - Hardware A/B:
    - `Customers_AB_HW`
    - `'Pct Customers AB HW'`
    - `Profit_AB_HW`
    - `'Pct Profit AB HW'`
  - CRE A/B:
    - `Customers_AB_CRE`
    - `'Pct Customers AB CRE'`
    - `Profit_AB_CRE`
    - `'Pct Profit AB CRE'`

- **Grade mix bar chart (hardware):**
  - Visual: 100% stacked column.
  - Axis: `ICP_grade_hardware`.
  - Values: `Customers`.

- **Grade mix bar chart (CRE):**
  - Visual: 100% stacked column.
  - Axis: `ICP_grade_cre`.
  - Values: `Customers_CRE`.

- **Industry performance bar chart (profit):**
  - Visual: column chart.
  - Axis: `Industry` (top N by profit).
  - Values: `Profit`.
  - Tooltip: `Customers`, `Customers_AB_HW`, `Customers_AB_CRE`.

This page is division-neutral but shows how much of the book is in A/B for both HW and CRE.

---

### 5.2 Hardware Territory A/B Focus

**Purpose:** Help HW reps and managers focus on the right A/B hardware accounts in each territory.

**Slicers:**

- `'icp_scored_accounts v1.5'[AM_Territory]`
- `'icp_scored_accounts v1.5'[am_sales_rep]`
- `'icp_scored_accounts v1.5'[Industry]`
- `'icp_scored_accounts v1.5'[ICP_grade_hardware]` (default filter to {A, B}).

**Visuals:**

- **Territory KPIs (cards):**
  - `Customers_AB_Territory_HW`
  - `Profit_AB_Territory_HW`
  - `'HW Revenue Share'` (optional)

- **Scatter: adoption vs relationship (HW view):**
  - X-axis: `Hardware_score`
  - Y-axis: `Software_score`
  - Size: `Profit`
  - Legend: `ICP_grade_hardware`
  - Tooltip: `Customer ID`, `Company Name`, `Industry`, `AM_Territory`, `printer_count`
  - Visual-level filter: `ICP_grade_hardware` ∈ {A, B}

- **A/B call list table (exportable):**
  - Columns (from `'icp_scored_accounts v1.5'`):
    - `Customer ID`, `Company Name`, `Industry`
    - `AM_Territory`, `am_sales_rep`
    - `ICP_grade_hardware`, `ICP_score_hardware`
    - `Hardware_score`, `adoption_assets`, `adoption_profit`
    - `printer_count`, `GP_Since_2023_Total`
    - Any playbook/tag columns (e.g., `call_to_action` / POV tags).
  - Sort by `ICP_score_hardware` then `Profit` descending.

This page is where HW sellers “live” when building their daily list.

---

### 5.3 CRE Territory A/B Focus

**Purpose:** Give CRE reps the same level of guidance as HW reps, but using CRE metrics.

**Slicers:**

- `'icp_scored_accounts v1.5'[CAD_Territory]` (or `AM_Territory` if that’s how CRE is structured).
- `'icp_scored_accounts v1.5'[cre_sales_rep]`
- `'icp_scored_accounts v1.5'[Industry]`
- `'icp_scored_accounts v1.5'[ICP_grade_cre]` (default filter to {A, B}).

**Visuals:**

- **Territory KPIs (cards):**
  - `Customers_AB_Territory_CRE`
  - `Profit_AB_Territory_CRE`
  - `'SW Revenue Share'` (optional proxy for CRE footprint).

- **Scatter: CRE adoption vs relationship:**
  - X-axis: `cre_adoption_assets` or `cre_adoption_profit`
  - Y-axis: `cre_relationship_profit`
  - Size: `Profit_CRE`
  - Legend: `ICP_grade_cre`
  - Tooltip: `Customer ID`, `Company Name`, `Industry`, `CAD_Territory`, `cre_sales_rep`
  - Visual-level filter: `ICP_grade_cre` ∈ {A, B}

- **CRE call list table (exportable):**
  - Columns:
    - `Customer ID`, `Company Name`, `Industry`
    - `CAD_Territory`, `cre_sales_rep`
    - `ICP_grade_cre`, `ICP_score_cre`
    - `Software_score`, `cre_adoption_assets`, `cre_adoption_profit`
    - `cre_relationship_profit`, `GP_Since_2023_Total`
    - CRE-specific playbook/tag columns (for example, “Cross-sell CAD”, “Expand Specialty Software”).

This page mirrors the HW A/B view, but from a CRE perspective.

---

### 5.4 Dual-division Call List Builder

**Purpose:** Provide a single page where sales ops or leaders can build/export call lists for both HW and CRE with consistent filters.

**Base tables:**

- `'icp_scored_accounts v1.5'` as the master.
- Optionally, join in the exported call lists CSV for historical comparison.

**Slicers:**

- `customer_segment`, `Industry`
- `AM_Territory`, `CAD_Territory`
- `am_sales_rep`, `cre_sales_rep`
- Grade filters:
  - `ICP_grade_hardware` (default A/B)
  - `ICP_grade_cre` (default A/B)

**Visuals:**

- **HW call list table:**
  - Filtered: `ICP_grade_hardware` ∈ {A, B}
  - Columns as in 5.2.

- **CRE call list table:**
  - Filtered: `ICP_grade_cre` ∈ {A, B}
  - Columns as in 5.3.

- **Optional presets (bookmarks):**
  - “HW expansion” – high `Hardware_score`, low `Software_score`.
  - “CRE expansion” – high `Software_score`, low `Hardware_score`.
  - “HW→CRE cross-sell” – strong HW profit, weak CRE signals.
  - “CRE→HW cross-sell” – strong CRE activity, few printers.

Each preset is just a bookmark capturing slicers/filters; they give users one-click entry points.

---

### 5.5 Similar Accounts / Look-alike Lab

**Purpose:** Recreate the Streamlit Look-alike Lab so sellers can pick a hero account, inspect its closest neighbors, and push those neighbors into a call list with HW/CRE-specific context.

**Relationships & Modeling Tips**

- Load rtifacts/account_neighbors.csv as icp_account_neighbors.
- Relationship 1 (anchor): icp_account_neighbors[account_id] -> 'icp_scored_accounts v1.5'[Customer ID].
- Relationship 2 (neighbor): use TREATAS/LOOKUPVALUE to pull neighbor metrics by 
eighbor_account_id, or create a second relationship via a duplicated table if you prefer a physical join.
- Keep 
eighbor_rank as Whole Number for sorting; treat similarity columns as decimals.

**Recommended Controls**

- Anchor slicer: concatenated Company Name / Customer ID field, filtered to A/B accounts in the current seller's AM/CAD territory (reuse the same slicers from the call list pages).
- Division toggle: disconnected table with values Hardware, CRE, Dual; use it in measures to swap between HW and CRE stats.
- Similarity threshold slider bound to icp_account_neighbors[sim_overall] (optional).

**Visual Blueprint**

1. **Anchor summary cards (HW + CRE):** Display ICP_score_*, GP_Since_2023_Total, whitespace_score, days_since_last_order, etc., for the selected anchor.
2. **Neighbor table:** Base visual = icp_account_neighbors. Add columns via LOOKUPVALUE or TREATAS to pull neighbor metrics: rank + similarity (
eighbor_rank, sim_overall, sim_numeric, sim_text, sim_als), company/industry/territory, HW stats (grades/scores, GP_Since_2023_Total, hw_share_12m, printer_count, whitespace_score), CRE stats (grades/scores, cre_relationship_profit, CRE_Training, sw_share_12m), and opportunity deltas such as Neighbor GP Gap, Training Gap, Whitespace Gap. Use conditional formatting to highlight the largest opportunity signals.
3. **Action buttons:** Buttons with bookmarks for "Send to Call List Builder" (captures current filters and navigates to the Call List page) and "Reset neighbors".
4. **Helper visuals:** Cards for neighbor count, average similarity, total whitespace, plus a bar chart showing neighbor counts by territory vs the anchor, and a KPI for "Orphan look-alikes" (neighbors with Dormant activity or long recency).

**Manager / Leadership Add-ons**

- Build a "Neighbor HQ" matrix grouped by AM/CAD territory with measures for Hero Accounts (A/B), Underpenetrated Neighbors, Potential Uplift (sum of whitespace gaps), and Orphan Look-alikes.
- Add a QoQ GP comparison visual for neighbors vs the overall territory to make QBR storytelling easy.

**Implementation Notes**

- docs/powerbi tmdl/powerbi_measures_clean.tmdl already includes neighbor-friendly measures; extend it with whitespace-gap measures and keep _HW/_CRE suffixes so division context is always obvious.
- Bookmark-driven navigation provides the "push to call list" experience with no custom code.
- Refresh the dataset after every scoring+neighbor run to keep Power BI aligned with the Streamlit Look-alike Lab.
### 5.6 Scoring Details & Validation (Optional Ops Page)

**Purpose:** Give analytics/ops a place to check schema, validation logs, and run status without leaving Power BI.

Suggested widgets:

- Cards:
  - Number of accounts (`Customers`)
  - Distinct industries, territories, reps
  - As-of date (from `as_of_date` column)
- Table: recent validation log entries (if you export them to a table or expose them via a simple Power Query).
- Simple matrix: counts of required columns present/missing, using `POWERBI_FIELD_REFERENCE_CLEAN` as a visual reference.

This page is more for you and ops than for reps, but it closes the loop between the pipeline and the dashboard.
    - `neighbor_industry`, `neighbor_territory`, `neighbor_segment`
    - Look up their scores via relationships or a measure.

---

## 5. Practical Modeling Tips

- **Use text for IDs** – Always set `Customer ID` to text to avoid losing leading zeros or formatting differences between systems.
- **Avoid bi-directional relationships** – Keep relationships single-direction (dimensions filter facts) to prevent ambiguous filter paths, especially once neighbors and call-lists are introduced.
- **Hide heavy helper columns** – Columns like `GP_T4Q_Total` or internal components (`cre_adoption_assets`) can be marked “hidden” in the model and exposed only through measures.
- **Tag measure tables** – Create a dedicated “Measures” display folder or a dummy measure table to keep navigation clean in Power BI.
- **Reuse existing TMDL** – `docs/powerbi tmdl/powerbi_measures.txt` already contains a richer measure set; you can import or adapt it rather than redefining everything by hand.

---

## 6. Next Small Improvements (Within This Repo)

Without introducing new data, the highest-impact, low-risk UI enhancements you can wire up in Power BI using the existing outputs are:

- Add **A/B coverage cards** and **grade-mix visuals** to highlight prioritization impact.
- Build a compact **Territory A/B dashboard** for managers with territory and rep slicers.
- Surface **call-to-action / playbook text** from the scored CSV (or call-list exports) directly in Power BI tables so sellers see recommended motions side by side with scores and profit.

Use this guide as the blueprint for your `.pbix` file; the schema and measures are aligned with the code and docs already in this repo.*** End Patch```} ***!


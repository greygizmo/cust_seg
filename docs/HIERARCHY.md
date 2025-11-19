
# Product Hierarchy & Division Mapping

This document outlines the current item hierarchy and division mapping logic used in the ICP scoring pipeline.

## Hierarchy Overview

The product hierarchy is structured as follows:

```
Item Internal ID → Item Rollup → Goal → Division → Super Division
```

### Database Validation Results

**Analysis Date:** 2025-11-19  
**Validation Status:** ✅ **VERIFIED**

| Level | Source Table | Mapping Type | Status |
|-------|--------------|--------------|--------|
| Item → Item Rollup | `items_category_limited` | 1:1 | ✅ Verified |
| Item Rollup → Goal | `analytics_product_tags` | 1:1 | ✅ Verified |

**Key Findings:**
- **70,764** unique items in `items_category_limited`
- Each item maps to exactly **ONE** `item_rollup`
- Each `item_rollup` maps to exactly **ONE** `Goal`
- **No duplicate GP risk** from multi-goal mappings

---

## Goal Taxonomy

The following 8 Goals are currently defined in `dbo.analytics_product_tags`:

| Goal | Division | Super Division | Description |
|------|----------|----------------|-------------|
| **CAD** | `cre` | Software | Computer-Aided Design software |
| **CPE** | `cpe` | Software | Customer Production Environment (future division) |
| **Specialty Software** | `cre` | Software | Specialized software solutions |
| **Training/Services** | `hardware` / `cre`* | Hybrid | Professional services and training |
| **Printers** | `hardware` | Hardware | 3D printing equipment |
| **Printer Accessorials** | `hardware` | Hardware | Printer-related accessories |
| **Scanners** | `hardware` | Hardware | 3D scanning equipment |
| **Geomagic** | `hardware` | Hardware | Geomagic software suite |

\* *Training/Services is a **hybrid category**. Most rollups are scored under Hardware, but specific items (e.g., "3DP Training", "Success Plan") are included in CRE relationship scoring.*

---

## Division Configurations

### Currently Active Divisions

The ICP scoring pipeline calculates scores for **three divisions**:

1. **Hardware Division** (`hardware`)
   - **Super Division:** Hardware
   - **Adoption Goals:** Printers, Printer Accessorials, Scanners, Geomagic, Training/Services
   - **Relationship Goals:** CAD, CPE, Specialty Software
   - **Component Weights:** Vertical (30%), Adoption (50%), Relationship (20%)

2. **CRE Division** (`cre`)
   - **Super Division:** Software
   - **Adoption Goals:** CAD, Specialty Software
   - **Relationship Goals:** Specialty Software, Training/Services (subset)
   - **Component Weights:** Vertical (25%), Adoption (45%), Relationship (30%)

3. **CPE Division** (`cpe`) — *Future-Ready Configuration*
   - **Super Division:** Software
   - **Adoption Goals:** CPE
   - **Relationship Goals:** CPE
   - **Component Weights:** Vertical (25%), Adoption (45%), Relationship (30%)
   - **Status:** ⚠️ Requires industry weights file (`cpe_industry_weights.json`)

### Division Configuration Files

Division configurations are defined in `src/icp/divisions.py` and can be overridden by JSON files in `artifacts/divisions/`.

---

## GP Accuracy & Deduplication Strategy

### Problem Addressed

**Original Risk:** If an `item_rollup` mapped to multiple Goals, summing GP by Goal would overcount profit (each line item would be counted multiple times).

### Validation & Solution

1. **Database Analysis Confirmed:** 1:1 mapping between `item_rollup` and `Goal` **eliminates this risk**.
2. **Code Safety Measures Implemented:**
   - `get_profit_since_2023_by_customer_rollup()` fetches profit without joining Goals
   - Feature engineering explicitly deduplicates by `[Customer ID, item_rollup]` before summing
   - Loader uses non-overlapping sources for `Profit_Since_2023_Total`

### Mathematical Soundness

✅ **GP calculations are now mathematically sound** because:
- Each transaction line is unique (enforced by `saleslog_detail`)
- Each `item_rollup` maps to exactly one Goal
- Deduplication logic ensures no double-counting even if future changes introduce complexity

---

## Implementation Notes

### To Add a New Division:

1. Define `DivisionConfig` in `src/icp/divisions.py`
2. Create industry weights file: `artifacts/weights/{division_key}_industry_weights.json`
3. Update `score_accounts.py` to calculate and rename scores for the new division
4. Add feature engineering logic if new adoption/relationship columns are needed
5. Update output schema in `schema.py`

### References:
- Configuration: `src/icp/divisions.py`
- Data Access: `src/icp/data_access.py`
- Feature Engineering: `src/icp/features/engineering.py`
- Scoring Logic: `src/icp/scoring.py`

# Customer Segmentation & ICP Scoring

## Recent Major Update: Hardware Adoption Score Algorithm

**As of July 2025, the Hardware Adoption Score logic has been significantly improved for more accurate and business-aligned ICP scoring.**

### 🚀 What Changed?
- **Weighted Printer Score:**
  - Big Box printers are now valued at 2x the weight of Small Box printers, reflecting their higher investment and engagement.
- **Comprehensive Revenue:**
  - The adoption score now includes both `Total Hardware Revenue` and `Total Consumable Revenue` for a complete picture of hardware engagement.
- **Percentile-Based Scaling:**
  - Both the weighted printer score and total hardware+consumable revenue are converted to percentile ranks (0-1) across all customers, ensuring fair comparison between different units.
- **Business Rules for True Adoption:**
  - **If a customer has zero printers AND zero hardware/consumable revenue, their adoption score is set to 0.0.**
  - **If a customer has revenue but no printers, their adoption score is capped at 0.4.**
  - **Only customers with actual printer investment can achieve high adoption scores.**
- **50/50 Weighting:**
  - The final adoption score is a 50/50 blend of the printer percentile and the revenue percentile (subject to the above business rules).

### 💡 Why This Matters
- **No more "phantom adopters":** Customers with no printers and no spend now get a true zero for adoption.
- **Revenue-only customers are recognized, but capped:** They can't outrank true hardware adopters.
- **Big Box investment is rewarded:** Customers with more significant hardware investment are prioritized.
- **ICP grades are now highly predictive of hardware sales potential.**

### 🔑 Impact on Sales Prioritization
- Hardware sales teams can now trust that high ICP grades reflect real, tangible hardware engagement.
- The adoption score is now the dominant factor in the optimized ICP model (50% weight), with industry and software relationship as supporting factors.

---

## Quick Start
... (existing content below remains unchanged) ... 
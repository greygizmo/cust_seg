<!-- Moved from documentation/INSTRUCTIONS_CSV_ENRICHMENT.md -->

# Industry Enrichment CSV Instructions

This guide documents how to prepare and use the optional industry enrichment CSV (`data/raw/TR - Industry Enrichment.csv`). When present, the scoring pipeline will apply enriched industry and sub‑industry classifications and include optional reasoning text.

Required columns:
- `Customer ID` (or alias columns mapped automatically: `ID`, `customer_id`)
- `Industry`
- `Industry Sub List`

Optional columns:
- `Reasoning` — free‑text rationale explaining enrichment choices
- `CRM Full Name (Customer)` — used as a fallback join key when needed

File placement:
- `data/raw/TR - Industry Enrichment.csv`

Validation and behavior:
- If the file is missing or columns are incomplete, the pipeline logs a warning and proceeds with original industry data.
- If both ID and CRM Full Name are available, the ID match takes precedence.


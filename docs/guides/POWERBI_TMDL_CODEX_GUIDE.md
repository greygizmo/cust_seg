
---
title: "Power BI TMDL — Codex Agent Usage Guide"
version: "1.0"
last_updated: "2025-11-17"
audience: "code-generation & automation agents"
purpose: "Generate valid, efficient TMDL for Power BI semantic models on the first try."
---

# Power BI TMDL — Codex Agent Usage Guide

> **Mission for the agent:** Produce *valid, deterministic, and minimal-diff* TMDL that round-trips cleanly, applies without errors, and adheres to modeling best practices. Prefer canonical templates and exact indentation over improvisation.

## Quick Start (Agent Checklist)

1. **Indentation is structure.** One indent per level (tab or 4 spaces). Never mix within a file. Mis-indentation = parse error.  
2. **Declarations:** `objectType Name` on its own line; properties one level deeper as `key: value`.  
3. **Default expression:** If the object has a primary expression, place it after `=` (e.g., measures, calc columns, calc items, partitions). For multi-line, start the expression on the next indented line.  
4. **Quoting:** Quote names with spaces/special chars using single quotes `'Name With Space'`. Escape single quotes with `''`.  
5. **Booleans:** `isHidden` alone implies `true`; otherwise use `isHidden: false`.  
6. **References:** Use object names (quoted if needed). Fully qualify columns where ambiguity is possible: `Table.'Column Name'`.  
7. **Ordering:** Let serializer manage `ref` lists; don’t hand-edit unless stabilizing object order.  
8. **Apply & refresh:** Applying TMDL changes updates metadata. Run a data **refresh** when queries/partitions change.  
9. **Determinism:** Keep property ordering stable, use display folders, and avoid gratuitous renames to minimize diffs.  
10. **Validate:** Use Desktop TMDL View or deserialize in code to catch syntax/semantic errors before deployment.

---

## Table of Contents

- [What is TMDL](#what-is-tmdl)
- [Folder & File Layout](#folder--file-layout)
- [Core Syntax Rules](#core-syntax-rules)
- [Keyword & Object Glossary](#keyword--object-glossary)
- [Canonical Templates](#canonical-templates)
  - [database.tmdl](#databasetmdl)
  - [model.tmdl](#modeltmdl)
  - [dataSources.tmdl](#datasourcestmdl)
  - [expressions.tmdl (M/Parameters)](#expressionstmdl-mparameters)
  - [tables/<Table>.tmdl (columns, measures, hierarchies, partitions)](#tablestabletmdl-columns-measures-hierarchies-partitions)
  - [relationships.tmdl](#relationshipstmdl)
  - [roles/<Role>.tmdl (RLS)](#rolesroletmdl-rls)
  - [perspectives/<Perspective>.tmdl](#perspectivesperspectivetmdl)
  - [Calculation Groups & Items](#calculation-groups--items)
- [Validation & Deployment](#validation--deployment)
- [Agent Patterns & Invariants](#agent-patterns--invariants)
- [Common Errors → Fixes](#common-errors--fixes)
- [Snippets Library](#snippets-library)
- [Appendix: Style Notes for DAX/M](#appendix-style-notes-for-daxm)

---

## What is TMDL

**Tabular Model Definition Language (TMDL)** is a human-readable, indentation-based language for defining Power BI / Analysis Services tabular models (aka semantic models). It represents all TOM (Tabular Object Model) objects as text files that are easy to diff, review, and version in source control.

**Two ways you’ll see TMDL:**
- **Model-as-files** (folder of `.tmdl` files) — ideal for Git, CI/CD, and programmatic generation.
- **Script blocks** in Desktop’s **TMDL View** — e.g., `createOrReplace` blocks to apply changes to a live/local model.

---

## Folder & File Layout

A typical TMDL folder for one model (dataset):

```
📁 MyModel-TMDL/
├── database.tmdl
├── model.tmdl
├── dataSources.tmdl
├── expressions.tmdl
├── relationships.tmdl
├── cultures/
│   ├── en-US.tmdl
│   └── <locale>.tmdl
├── perspectives/
│   └── <PerspectiveName>.tmdl
├── roles/
│   └── <RoleName>.tmdl
└── tables/
    ├── <TableA>.tmdl
    ├── <TableB>.tmdl
    ├── icp_scored_accounts.tmdl
    ├── icp_account_playbooks.tmdl
    ├── account_neighbors.tmdl
    └── ...
```

> Agent tip: Keep one table per file. Avoid massive monoliths. Smaller files yield clearer diffs.

---

## Core Syntax Rules

**1) Declarations**
```tmdl
table Product
    column 'Product Key'
        dataType: int64
        isKey
        sourceColumn: ProductKey
```

**2) Properties**: `name: value` under the object’s indent. Booleans can be shorthand (`isHidden`).

**3) Default expression via `=`** (measures, calc columns, calc items, partitions, tablePermission):
```tmdl
measure "Gross Margin %" =
        VAR revenue = [Total Sales]
        VAR cost = [Total Cost]
        RETURN DIVIDE(revenue - cost, revenue)
    formatString: "0.00%;-0.00%;0.00%"
```

**4) Multi-line expressions:**
- Place the expression on the next line after `=` at +1 indent.
- Keep all expression lines at that indent.
- Dedent to continue with properties.

**5) Quoting names:**
- Quote when the name has spaces/special chars: `'Category Order'`.
- Escape single quotes via `''`: `'Bob O''Reilly'`.

**6) Comments → Descriptions:**
- Use `///` lines directly above an object to populate its Description.
```tmdl
/// Total sales in USD after returns
measure "Total Sales" = SUMX(Sales, Sales[Qty] * Sales[Net Price])
```

**7) `ref` entries (ordering hints):**
- You may see `ref table X` etc. in `model.tmdl` to preserve object order.
- Usually generated/maintained by the serializer; agents seldom need to hand-write these.

**8) Language values:**
- **DAX** inside measures, calc columns, calc items, tablePermission filters.
- **M** inside import partitions and shared expressions/parameters.

---

## Keyword & Object Glossary

| Keyword / Block         | Purpose (Agent summary)                                      |
|---                      |---                                                           |
| `database`              | Dataset container (name, compatibility, IDs).               |
| `model`                 | Model-wide settings, culture, annotations, refs.            |
| `dataSource`            | Connection metadata to sources (provider, conn string, etc).|
| `expression`            | Shared Power Query (M) expression or parameter.             |
| `table`                 | A table (columns, measures, hierarchies, partitions).       |
| `column`                | Column in a table; calc column uses `=` DAX expression.     |
| `measure`               | DAX measure with optional properties (format, folder).      |
| `hierarchy` / `level`   | Hierarchy and its levels (each level references a column).  |
| `partition`             | Data retrieval for a table (Import M, DirectQuery, DAX).    |
| `relationship`          | Relationship (fromColumn → toColumn, active, cross-filter). |
| `role` / `tablePermission` | RLS role and its DAX row filter(s).                    |
| `perspective`           | Subset view: which fields/measures are visible.             |
| `calculationGroup`      | Special table with `calculationItem`s for reusable logic.   |
| `calculationItem`       | DAX transformation applied to `SELECTEDMEASURE()`.          |
| `annotation`            | Extra metadata key/value.                                    |
| `culture`               | Translations for captions/descriptions by locale.           |

> Common property values: `dataType: int64|string|double|decimal|boolean|dateTime|date|time|dateTimeOffset`, `summarizeBy: none|sum|min|max|average|count|distinctCount`, `isHidden`, `isKey`, `formatString`, `displayFolder`, `sortByColumn`, `crossFilteringBehavior: SingleDirection|BothDirections`, `isActive: true|false`.

---

## Canonical Templates

### `database.tmdl`
```tmdl
database <DatasetName>
    compatibilityLevel: 1567
    // Optional system properties like ID or mashup may be present; usually do not hand-edit.
```

### `model.tmdl`
```tmdl
model Model
    culture: en-US
    // Optional: annotations, default summarization, formatting culture, etc.

    // Ordering hints (optional, typically auto-managed):
    // ref table Calendar
    // ref table Product
    // ref table Sales
```

### `dataSources.tmdl`
```tmdl
dataSource SqlDatabase
    provider: "SQL"
    connectionString: "Provider=SQLNCLI11;Data Source=<SERVER>;Initial Catalog=<DB>;Integrated Security=true"
    impersonationMode: Implicit
    name: "<DB>"
```

### `expressions.tmdl` (M/Parameters)
```tmdl
// Parameter example
expression ServerName = "SQLPROD01" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]

// Shared M query example
expression DateQuery =
        let
            Source = Sql.Database(ServerName, "SalesDB"),
            Dates = Source{[Schema="dbo", Item="DimDate"]}[Data]
        in
            Dates
```

### `tables/<Table>.tmdl` (columns, measures, hierarchies, partitions)

**Columns & Hierarchies**
```tmdl
table Product

    column 'Product Key'
        dataType: int64
        isKey
        sourceColumn: ProductKey
        summarizeBy: none

    column Name
        dataType: string
        sourceColumn: ProductName

    column Category
        dataType: string
        sourceColumn: Category
        sortByColumn: 'Category Order'

    column 'Category Order'
        dataType: int64
        isHidden
        sourceColumn: CategoryOrder

    // Calculated column
    column FullLabel = Product[Category] & " - " & Product[Name]
        dataType: string
        formatString: "@"
        isHidden

    hierarchy 'Product Hierarchy'
        level Category
            column: Category
        level Product
            column: Name
```

**Measures**
```tmdl
table Sales

    /// Total revenue from sales (in USD)
    measure "Total Sales" = SUMX(Sales, Sales[Quantity] * Sales[Net Price])
        formatString: "$ #,##0"
        displayFolder: "Revenue Metrics"

    measure "Total Cost" = SUMX(Sales, Sales[Quantity] * Sales[Unit Cost])
        formatString: "$ #,##0"
        displayFolder: "Revenue Metrics"

    measure "Gross Margin %" =
            VAR revenue = [Total Sales]
            VAR cost = [Total Cost]
            RETURN IF(revenue <> 0, (revenue - cost) / revenue)
        formatString: "0.00%;-0.00%;0.00%"
        displayFolder: "Revenue Metrics"
        description: "Gross margin percentage"
```

**Partitions**

_Import (M) partition_
```tmdl
table Sales

    partition 'Sales-Partition' = m
        mode: import
        source =
            let
                Source = Sql.Database(ServerName, "SalesDB"),
                SalesTbl = Source{[Schema="dbo", Item="FactSales"]}[Data],
                Cleaned = Table.RemoveColumns(SalesTbl, {"Unused1","Unused2"})
            in
                Cleaned
```

_DirectQuery (native SQL optional)_
```tmdl
table Sales_DQ

    partition 'Sales-DQ'
        mode: DirectQuery
        // Optional native query:
        // query: SELECT * FROM dbo.FactSales
```

_Calculated table (DAX)_
```tmdl
table Calendar

    partition 'Calendar-Calc' =
            CALENDAR(DATE(2020,1,1), DATE(2030,12,31))
        mode: import
```

### `relationships.tmdl`
```tmdl
relationship 12345678-90ab-cdef-1234-567890abcdef
    fromColumn: Sales.'Product Key'
    toColumn: Product.'Product Key'
    crossFilteringBehavior: SingleDirection
    isActive: true
```

### `roles/<Role>.tmdl` (RLS)
```tmdl
role StoreManagers
    modelPermission: read

    tablePermission Store = 'Store'[Store Code] IN { 1, 10, 20, 30 }
```

### `perspectives/<Perspective>.tmdl`
```tmdl
perspective SalesPerspective
    perspectiveTable Sales
        perspectiveColumn Quantity
        perspectiveColumn 'Net Price'
        perspectiveMeasure "Total Sales"
    perspectiveTable Product
        perspectiveColumn Name
        perspectiveColumn Category
```

### Calculation Groups & Items
```tmdl
// Script-style example (Desktop TMDL View supports createOrReplace)
createOrReplace

    table 'Time Intelligence'
        calculationGroup
            precedence: 1

            calculationItem YTD =
                    CALCULATE(SELECTEDMEASURE(), DATESYTD('Calendar'[Date]))

            calculationItem QTD =
                    CALCULATE(SELECTEDMEASURE(), DATESQTD('Calendar'[Date]))

            calculationItem MTD =
                    CALCULATE(SELECTEDMEASURE(), DATESMTD('Calendar'[Date]))

            calculationItem Current = SELECTEDMEASURE()

        column 'Show As'
            dataType: string
            sourceColumn: Name
            sortByColumn: Ordinal

        column Ordinal
            dataType: int64
            formatString: 0
            isHidden
```

---

## Validation & Deployment

### Validate in Power BI Desktop (TMDL View)
- Enable TMDL View in Options (if required by your version).
- Author scripts, then **Apply**. Errors show with line/column; fix and re-apply.
- Applying changes updates metadata; **refresh data** if partitions/queries changed.

### Programmatic (C# / TOM with TMDL Serializer)
```csharp
using Microsoft.AnalysisServices.Tabular; // AMO/TOM
// Serialize existing dataset to folder
TmdlSerializer.SerializeDatabaseToFolder(database, @"C:\out\MyModel-TMDL");

// Validate (syntax/semantic) by deserializing
var model = TmdlSerializer.DeserializeModelFromFolder(@"C:\out\MyModel-TMDL");

// Deploy changes to target
using (var server = new Server())
{
    server.Connect("DataSource=powerbi://api.powerbi.com/v1.0/myorg/<WorkspaceName>"); // XMLA endpoint
    var db = server.Databases[model.Database.ID];
    model.CopyTo(db.Model);
    db.Model.SaveChanges(); // apply
}
```

### CI/CD Ideas (conceptual)
- Store the TMDL folder in Git.
- Build step: deserialize to validate; fail the build on error.
- Release step: connect via XMLA and apply; then trigger dataset refresh if needed.

---

## Agent Patterns & Invariants

1. **Stable Names:** Don’t rename objects unless explicitly requested; renames inflate diffs and can break dependencies.  
2. **Quote Policy:** Quote any name with whitespace or special chars. Be consistent.  
3. **Display Folders:** Group related measures (e.g., `"Revenue Metrics"`) to keep field lists clean.  
4. **Sort-by Columns:** When creating label columns (e.g., MonthName), add a corresponding numeric sort key and set `sortByColumn`.  
5. **Keys & Relationships:** Mark PK columns with `isKey`; use `summarizeBy: none`. Always fully qualify relationship endpoints.  
6. **Calc Groups:** Prefer calc groups over duplicating YTD/QTD/MTD measures. Add an Ordinal column and set `sortByColumn`.  
7. **RLS:** Keep filters simple and table-scoped with `tablePermission`. Validate DAX predicates with sample users if possible.  
8. **Partitions:** For Import, ensure M scripts are source-parameterized (e.g., `ServerName`) to support environment promotion.  
9. **Error Surfacing:** Emit minimal but actionable error messages; include file and line numbers when possible.  
10. **Idempotence:** Generate TMDL that can be applied repeatedly without drift (avoid volatile ordering; let serializer manage `ref`).

### SQL-First Patterns
When migrating from CSV to SQL sources:
1.  **M Query Updates**: Replace `Csv.Document` with `Sql.Database`.
2.  **Type Mapping**: SQL sources have typed columns, so `Table.TransformColumnTypes` may be redundant but is often kept for explicit safety in Power BI.
3.  **Server/DB Parameters**: Use variables or parameters for `ServerName` and `DatabaseName` to facilitate environment switching (Dev/Prod).
4.  **Direct Query vs Import**: Default to Import for performance unless real-time data is strictly required.
5.  **TMDL Partitions**: Ensure the `partition` block in TMDL reflects the new M script.

---

## Common Errors → Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| “Unexpected indent / inconsistent indent” | Mixed tabs/spaces; wrong level | Use one indent style; align properties/blocks exactly one level under their owner. |
| “Unquoted name” | Space/special char in object name | Wrap in single quotes `'Name With Space'`; escape `'` as `''`. |
| “Unknown enum/value” | Invalid `dataType`, `summarizeBy`, etc. | Use known values (e.g., `int64`, `string`, `double`; `summarizeBy: none`). |
| “Object already exists” | Duplicate names (same scope) | Ensure uniqueness per scope; consolidate or rename intentionally once. |
| “Unknown reference” | `sortByColumn` / relationship points to missing column | Create the target first; fully qualify; correct spelling/quotes. |
| “RLS filter invalid” | DAX expression error | Validate DAX; keep to boolean-returning filters; qualify columns. |
| “Partition invalid” | M/DAX query error | Validate M/DAX separately; refresh after apply; parameterize connections. |

---

## Snippets Library

**Create a Date dimension (calc table + hierarchy):**
```tmdl
table 'Date'

    partition 'Date-Calc' =
            CALENDAR(DATE(2015,1,1), DATE(2035,12,31))
        mode: import

    column 'Date'
        dataType: date
        summarizeBy: none

    column 'Year' = YEAR('Date'[Date])
        dataType: int64
        summarizeBy: none

    column 'Month Number' = MONTH('Date'[Date])
        dataType: int64
        summarizeBy: none

    column 'Month' = FORMAT('Date'[Date], "MMM")
        dataType: string
        sortByColumn: 'Month Number'

    hierarchy 'Calendar'
        level Year
            column: 'Year'
        level Month
            column: 'Month'
        level Day
            column: 'Date'
```

**Time Intelligence calculation group (YTD/QTD/MTD/Current):**
```tmdl
createOrReplace

    table 'Time Intelligence'
        calculationGroup
            precedence: 1

            calculationItem YTD =
                    CALCULATE(SELECTEDMEASURE(), DATESYTD('Date'[Date]))

            calculationItem QTD =
                    CALCULATE(SELECTEDMEASURE(), DATESQTD('Date'[Date]))

            calculationItem MTD =
                    CALCULATE(SELECTEDMEASURE(), DATESMTD('Date'[Date]))

            calculationItem Current = SELECTEDMEASURE()

        column 'Show As'
            dataType: string
            sourceColumn: Name
            sortByColumn: Ordinal

        column Ordinal
            dataType: int64
            isHidden
```

**RLS Role (region-limited access):**
```tmdl
role RegionUS
    modelPermission: read

    tablePermission Sales = Sales[Region] = "US"
```

**Import partition using shared M query:**
```tmdl
table DimDate

    partition 'DimDate-Import' = m
        mode: import
        source = DateQuery
```

**One-to-many relationship:**
```tmdl
relationship 0d1cfe2f-4aa1-4cd2-9a2b-6f5a66f2a0d1
    fromColumn: Sales.'Customer Key'
    toColumn: Customer.'Customer Key'
    crossFilteringBehavior: SingleDirection
    isActive: true
```

---

## Appendix: Style Notes for DAX/M

- **DAX:** Prefer `VAR/RETURN` for non-trivial measures; always provide `formatString` for user-facing measures; avoid iterators unless necessary; use `DIVIDE(x,y)` over `x / y`.  
- **M (Power Query):** Keep `let/in` readable; parameterize server/database; avoid hard-coded credentials; remove unused columns early; prefer query folding-friendly operations.

---

## End of Guide

This document is structured for deterministic generation. Agents should emit exactly these patterns, preserving indentation and quoting rules, and should validate by deserializing or applying in Desktop before promotion.

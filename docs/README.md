# System Documentation

This directory contains Mermaid diagrams and reference docs for the ICP scoring and neighbors pipeline.

## Documentation Overview

The documentation is organized into Mermaid diagrams and topical guides.

### System Architecture
- 01-overall-architecture.md — System overview and data flow
- 07-data-flow-dependencies.md — File relationships and dependencies
- 08-component-interaction.md — Python module interactions
- 09-file-relationships.md — Import/export relationships

### Data Processing and Neighbors
- 02-data-processing-pipeline.md — Scoring workflow and exact neighbors stage

### Scoring System
- 03-scoring-methodology.md — 4-component ICP scoring (division-aware, no legacy fallbacks)
- 04-weight-optimization.md — Weight tuning with Optuna (future GP labels, stability and lift)
- 06-industry-scoring.md — Data-driven industry weights (Empirical-Bayes + strategic blend)

### Guides
- ../METRICS_OVERVIEW.md — metrics and artifacts, config keys, CLI
- ../guides/POWERBI_FIELD_REFERENCE_CLEAN.md — field dictionary for scored accounts and neighbors
- ../guides/INSTRUCTIONS_CSV_ENRICHMENT.md — optional industry enrichment CSV

## Viewing the diagrams
Open the markdown files in an editor or Mermaid-enabled renderer.

## Chart Categories

### System Architecture
Overview of how components connect, including data flow and module interactions.

### Data Processing
Step-by-step pipeline from Azure SQL inputs to scored accounts and neighbors.

### Scoring System
Component scoring, weight optimization, and industry weighting.

## Mermaid Diagrams

Each chart includes:
- Visual diagrams (flowcharts and architecture)
- Explanations of each component
- Technical specifications (parameters, formulas, business rules)
- Integration points between components

The diagrams are designed to be technically accurate and approachable for both technical and business readers.

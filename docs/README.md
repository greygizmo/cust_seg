# System Documentation

This directory contains Mermaid diagrams and reference docs for the ICP Scoring and Neighbors pipeline.

##  Documentation Overview

The documentation is organized into Mermaid diagrams and topical guides:

###  System Architecture (3 charts)
- **01-overall-architecture.md**: Complete system overview and data flow
- **07-data-flow-dependencies.md**: File relationships and dependencies
- **08-component-interaction.md**: Python module interactions
- **09-file-relationships.md**: Import/export relationships

###  Data Processing (and Neighbors)
- **02-data-processing-pipeline.md**: 8-stage scoring workflow plus Stage 9 exact neighbors

###  Scoring System (3 charts)
- **03-scoring-methodology.md**: 4-component ICP scoring system
- **04-weight-optimization.md**: ML optimization with Optuna
- **06-industry-scoring.md**: Data-driven industry weights

###  Guides
- `../METRICS_OVERVIEW.md` — metrics and artifacts, config keys, CLI
- `../guides/INSTRUCTIONS_CSV_ENRICHMENT.md` — optional industry enrichment CSV

##  Viewing the diagrams
Open the markdown files in this folder in your editor or a Mermaid-enabled renderer.

##  Chart Categories

### System Architecture
Complete technical overview of how all components work together, including data flow, dependencies, and interactions between Python modules.

### Data Processing
Detailed pipeline showing how raw data is transformed into scored accounts, plus an exact, blockwise neighbors stage.

### Scoring System
Deep dive into the 4-component scoring methodology, ML optimization process, and industry-specific weight calculations.

### User Interface
Documentation of the dashboard user experience, interaction flows, and real-time analysis capabilities.

##  Mermaid Diagrams

Each chart uses Mermaid syntax and includes:
- **Visual diagrams**: Clear flowcharts and architecture diagrams
- **Detailed explanations**: Comprehensive descriptions of each component
- **Technical specifications**: Parameters, formulas, and business rules
- **Integration points**: How components connect and interact

The diagrams are designed to be both technically accurate and accessible to different audiences, from data scientists to business stakeholders.





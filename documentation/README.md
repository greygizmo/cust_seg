# System Documentation

This directory contains comprehensive Mermaid diagrams documenting the entire ICP Scoring System for GoEngineer.

##  Documentation Overview

The documentation is organized into **9 detailed Mermaid diagrams** covering all aspects of the system:

###  System Architecture (3 charts)
- **01-overall-architecture.md**: Complete system overview and data flow
- **07-data-flow-dependencies.md**: File relationships and dependencies
- **08-component-interaction.md**: Python module interactions
- **09-file-relationships.md**: Import/export relationships

###  Data Processing (1 chart)
- **02-data-processing-pipeline.md**: 8-stage data processing workflow

###  Scoring System (3 charts)
- **03-scoring-methodology.md**: 4-component ICP scoring system
- **04-weight-optimization.md**: ML optimization with Optuna
- **06-industry-scoring.md**: Data-driven industry weights

###  User Interface (1 chart)
- **05-dashboard-workflow.md**: User interaction and experience

##  Accessing Documentation in Dashboard

The documentation is now integrated into the Streamlit dashboard:

1. **Navigate to the dashboard**: Run `streamlit run streamlit_icp_dashboard.py`
2. **Use the sidebar navigation**: Select " System Documentation"
3. **Choose a category**: Select from the 4 main categories
4. **Browse charts**: Expand individual charts to view Mermaid diagrams and explanations

##  Additional Information

The documentation is also available as a separate page called " Scoring Details" which provides:
- Detailed explanations of each scoring component
- Current weight values and optimization status
- Mathematical formulas and business rules
- Grade assignment criteria

##  Chart Categories

### System Architecture
Complete technical overview of how all components work together, including data flow, dependencies, and interactions between Python modules.

### Data Processing
Detailed 8-stage pipeline showing how raw data is transformed into actionable insights, from data loading to final score calculation.

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





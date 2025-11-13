# Dashboard Workflow & User Experience

```mermaid
graph TD
    %% Define the dashboard user journey
    subgraph "Dashboard Initialization"
        DataLoading[Load Scored Data<br/>data/processed/icp_scored_accounts.csv<br/>Cached for performance]
        WeightLoading[Load Optimized Weights
artifacts/weights/optimized_weights.json<br/>Fallback to DEFAULT_WEIGHTS if missing]
        ConfigSetup[Initialize Dashboard Config<br/>- Default segment thresholds<br/>- Session state management<br/>- UI component defaults]
    end

    subgraph "Main Dashboard Display"
        HeaderSection[Header Section<br/> ICP SCORING DASHBOARD + Call List Builder
Shows current segment filter]
        MetricCards[Key Metrics Cards<br/>6 Interactive Metric Cards:<br/>- Total Customers<br/>- Average ICP Score<br/>- High-Value Customers (70)<br/>- Total Revenue<br/>- High-Value GP<br/>- High-Value Percentage]
    end

    subgraph "Customer Segmentation Controls"
        SegmentConfig[Configure Segments<br/> Configure Customer Segments<br/>- Small Business: $0-$100M<br/>- Mid-Market: $100M-$1B<br/>- Large Enterprise: >$1B<br/>- Customizable thresholds]
        SegmentSelector[Select Active Segment<br/>All Segments / Small Business /<br/>Mid-Market / Large Enterprise<br/>Filters entire dashboard view]
        DataFiltering[Apply Segment Filter<br/>Filter DataFrame based on selection<br/>Update all metrics and charts]
    end

    subgraph "Interactive Weight Controls"
        SidebarControls[Sidebar Weight Controls<br/> ICP Scoring Controls<br/>Displays optimization status]
        WeightSliders[Real-time Weight Sliders<br/>- Vertical Score Weight (0.0-1.0)<br/>- Size Score Weight (0.0-1.0)<br/>- Adoption Score Weight (0.0-1.0)<br/>- Relationship Score Weight (0.0-1.0)<br/>Auto-normalize to sum = 1.0]
        OptimizationStatus[Optimization Status Display<br/> Optimized Weights Active<br/>Shows trials, lambda parameter<br/> Using Default Weights<br/>If optimization not run]
    end

    subgraph "Real-time Score Recalculation"
        WeightChangeDetection[Detect Weight Changes<br/>Monitor slider changes<br/>Trigger recalculation]
        ScoreRecalculation[Recalculate All Scores<br/>Using new weights<br/>Call calculate_scores() function<br/>Update ICP scores and grades]
        DataUpdate[Update Dashboard Data<br/>Refresh metrics, charts, tables<br/>Maintain segment filtering]
    end

    subgraph "Interactive Visualizations"
        subgraph "Primary Charts Row 1"
            ICPDistribution[ICP Score Distribution<br/>Histogram + Box Plot<br/>Shows score spread and percentiles<br/>Updates with weight changes]
            GradeDistribution[Customer Grade Distribution<br/>Pie Chart (A-F)<br/>Shows grade distribution<br/>Color-coded by grade]
        end

        subgraph "Primary Charts Row 2"
            WeightDistribution[Weight Distribution Radar<br/>5-axis radar chart<br/>Shows current weight allocation<br/>Visual weight comparison]
            ScoreByIndustry[Average Score by Industry<br/>Horizontal bar chart<br/>Top 15 industries by score<br/>Shows customer count per industry]
        end

        subgraph "Additional Analytics"
            SegmentComparison[Segment Comparison Charts<br/>When "All Segments" selected<br/>- Average score by segment<br/>- Customer count by segment<br/>- Revenue by segment<br/>- A-grade distribution by segment]
        end
    end

    subgraph "Data Export & Details"
        TopCustomersTable[Top Scoring Customers Table
Columns: Company, Industry,
ICP_score, ICP_grade, Component Scores
Sortable and searchable]
        CSVExport[Export Current View
Download [Segment] Scores (CSV)
Filtered data with current weights
Includes all scores and metadata]
        CallListBuilder[Call List Builder
Filters: segment, industry, adoption/relationship bands
Toggles: revenue-only, heavy fleet, A/B only
Export CSV to reports/call_lists/]
    end

    subgraph "User Interaction Flow"
        InitialLoad[User Opens Dashboard<br/>Streamlit app loads<br/>Data and weights loaded]
        SegmentSelection[User Selects Segment<br/>Updates all views<br/>Maintains weight settings]
        WeightAdjustment[User Adjusts Weights<br/>Real-time recalculation<br/>Immediate visual feedback]
        ChartInteraction[User Explores Charts<br/>Interactive Plotly charts<br/>Hover tooltips, zoom, pan]
        DataExport[User Exports Data<br/>CSV download with current settings<br/>For further analysis or reporting]
    end

    %% Define user flow connections
    InitialLoad --> HeaderSection
    InitialLoad --> MetricCards
    InitialLoad --> SidebarControls

    HeaderSection --> SegmentConfig
    SegmentConfig --> SegmentSelector
    SegmentSelector --> DataFiltering

    DataFiltering --> MetricCards
    DataFiltering --> ICPDistribution
    DataFiltering --> GradeDistribution
    DataFiltering --> WeightDistribution
    DataFiltering --> ScoreByIndustry
    DataFiltering --> TopCustomersTable

    SidebarControls --> WeightSliders
    WeightSliders --> OptimizationStatus

    WeightSliders --> WeightChangeDetection
    WeightChangeDetection --> ScoreRecalculation
    ScoreRecalculation --> DataUpdate

    DataUpdate --> MetricCards
    DataUpdate --> ICPDistribution
    DataUpdate --> GradeDistribution
    DataUpdate --> WeightDistribution
    DataUpdate --> ScoreByIndustry
    DataUpdate --> TopCustomersTable
    DataUpdate --> CallListBuilder

    MetricCards --> ChartInteraction
    ICPDistribution --> ChartInteraction
    GradeDistribution --> ChartInteraction
    WeightDistribution --> ChartInteraction
    ScoreByIndustry --> ChartInteraction

    TopCustomersTable --> DataExport
    DataExport --> CSVExport

    %% Style definitions
    classDef initialization fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef mainDisplay fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef segmentation fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef controls fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef calculation fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef visualization fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef export fill:#f9fbe7,stroke:#689f38,stroke-width:2px
    classDef userFlow fill:#ffecb3,stroke:#f57c00,stroke-width:2px

    %% Apply styles
    class DataLoading,WeightLoading,ConfigSetup initialization
    class HeaderSection,MetricCards mainDisplay
    class SegmentConfig,SegmentSelector,DataFiltering segmentation
    class SidebarControls,WeightSliders,OptimizationStatus controls
    class WeightChangeDetection,ScoreRecalculation,DataUpdate calculation
    class ICPDistribution,GradeDistribution,WeightDistribution,ScoreByIndustry,SegmentComparison visualization
    class TopCustomersTable,CSVExport,CallListBuilder export
    class InitialLoad,SegmentSelection,WeightAdjustment,ChartInteraction,DataExport userFlow
```

## Dashboard Workflow & User Experience

The Streamlit dashboard provides an interactive, real-time interface for exploring and analyzing ICP (Ideal Customer Profile) scores. It enables users to understand customer segmentation, adjust scoring weights, and export results for further analysis.

### Dashboard Features:

#### 1. Real-time Interactivity
- **Dynamic Weight Adjustment**: Four sliders control the importance of each scoring component
- **Instant Recalculation**: All scores, metrics, and charts update immediately
- **Segment Filtering**: Filter the entire dashboard by customer segment

#### 2. Key Metrics Overview
- **Total Customers**: Count in current segment
- **Average ICP Score**: Mean score (0-100 scale)
- **High-Value Customers**: Count with score 70
- **Total Revenue**: Hardware + consumable revenue
- **High-Value Revenue**: Revenue from high-value customers
- **High-Value Percentage**: Percentage of customers scoring 70

#### 3. Interactive Visualizations
- **ICP Distribution**: Histogram showing score spread and statistical markers
- **Grade Distribution**: Pie chart showing A-F grade breakdown
- **Weight Radar**: Visual representation of current weight allocation
- **Industry Analysis**: Average scores by top 15 industries
- **Segment Comparison**: When viewing all segments, shows performance by segment

#### 4. Data Export
- **Top 100 Customers**: Table showing highest-scoring customers with all details
- **CSV Export**: Download current filtered dataset with all scores and metadata
- **Segment-Specific**: Export filename includes current segment filter

### User Journey:

1. **Initial Load**: Dashboard loads with optimized weights (if available) or defaults
2. **Segment Selection**: Choose to view all customers or specific segments
3. **Weight Adjustment**: Modify component weights using sliders
4. **Real-time Feedback**: See immediate impact on all metrics and charts
5. **Data Exploration**: Interact with charts, hover for details, zoom/pan
6. **Export Results**: Download data for further analysis or reporting

### Technical Features:
- **Caching**: Data loaded once for performance
- **Session State**: Maintains user selections across interactions
- **Responsive Design**: Works on different screen sizes
- **Error Handling**: Graceful fallbacks for missing data
- **Optimization Status**: Shows whether ML-optimized weights are active

### Business Value:
- **Strategic Planning**: Understand customer value distribution
- **Sales Prioritization**: Identify high-value accounts for focused efforts
- **Weight Sensitivity**: Test different scoring strategies
- **Segment Analysis**: Compare performance across customer segments
- **Data-Driven Decisions**: Export results for integration with other systems





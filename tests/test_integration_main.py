from unittest.mock import patch, MagicMock
import pandas as pd
from icp.cli.score_accounts import main, CLI_OPTS

@patch('icp.cli.score_accounts.da')
@patch('icp.cli.score_accounts.assemble_master_from_db')
@patch('icp.cli.score_accounts.engineer_features')
@patch('icp.cli.score_accounts.calculate_scores')
@patch('icp.cli.score_accounts.enrich_with_list_builder_features')
@patch('icp.cli.score_accounts.build_visuals')
@patch('icp.cli.score_accounts.build_neighbors')
@patch('icp.cli.score_accounts.pd.DataFrame.to_csv')
@patch('icp.cli.score_accounts.validate_master')
@patch('icp.cli.score_accounts.pd.DataFrame.to_sql')
def test_main_integration(
    mock_to_sql, mock_validate, mock_to_csv, mock_neighbors, mock_visuals, 
    mock_enrich, mock_calculate, mock_engineer, mock_assemble, mock_da
):
    # Setup mocks
    mock_assemble.return_value = pd.DataFrame({'Customer ID': ['1'], 'Industry': ['Tech']})
    mock_validate.return_value = pd.DataFrame({'Customer ID': ['1'], 'Industry': ['Tech']}) # Pass through
    
    # Mock engineer_features output (Hardware pass)
    mock_engineer.return_value = pd.DataFrame({
        'Customer ID': ['1'], 
        'ICP_score': [80], 
        'ICP_grade': ['B'],
        'adoption_score': [0.5],
        'relationship_score': [0.5],
        'vertical_score': [0.5],
        'ICP_score_raw': [80],
        'printer_count': [1]
    })
    
    # Mock calculate_scores output (CRE pass)
    mock_calculate.return_value = pd.DataFrame({
        'Customer ID': ['1'], 
        'ICP_score': [70], 
        'ICP_grade': ['C'],
        'adoption_score': [0.4],
        'relationship_score': [0.4],
        'vertical_score': [0.4],
        'ICP_score_raw': [70],
        'printer_count': [1],
        # Include renamed Hardware scores as they are passed through
        'Hardware_ICP_Score': [80],
        'Hardware_ICP_Grade': ['B'],
        'Hardware_Adoption_Score': [0.5],
        'Hardware_Relationship_Score': [0.5],
        'Hardware_Vertical_Score': [0.5],
        'Hardware_ICP_Score_Raw': [80],
    })

    mock_enrich.return_value = pd.DataFrame({
        'Customer ID': ['1'], 
        'Hardware_ICP_Score': [80], 
        'CRE_ICP_Score': [70],
        'printer_count': [1],
        'GP_Printers': [100],
        'GP_Specialty Software': [50]
    })
    
    # Mock engine
    mock_da.get_engine.return_value = MagicMock()
    mock_da.get_sales_detail_since_2022.return_value = pd.DataFrame()
    
    # Set CLI opts
    CLI_OPTS['division'] = 'hardware'
    CLI_OPTS['skip_neighbors'] = True
    CLI_OPTS['skip_visuals'] = True
    CLI_OPTS['out_path'] = 'test_output.csv'
    
    # Run main
    main()
    
    # Verify calls
    mock_assemble.assert_called_once()
    mock_validate.assert_called_once()
    mock_engineer.assert_called_once()
    # Called for CRE and CPE
    assert mock_calculate.call_count >= 2
    mock_enrich.assert_called_once()
    mock_to_csv.assert_called_once()
    # Visuals and neighbors skipped
    mock_visuals.assert_not_called()
    mock_neighbors.assert_not_called()

@patch('icp.cli.score_accounts.da')
@patch('icp.cli.score_accounts.assemble_master_from_db')
@patch('icp.cli.score_accounts.engineer_features')
@patch('icp.cli.score_accounts.calculate_scores')
@patch('icp.cli.score_accounts.enrich_with_list_builder_features')
@patch('icp.cli.score_accounts.build_visuals')
@patch('icp.cli.score_accounts.build_neighbors')
@patch('icp.cli.score_accounts.validate_master')
@patch('icp.cli.score_accounts.pd.DataFrame.to_csv')
def test_main_integration_with_neighbors(
    mock_to_csv, mock_validate, mock_neighbors, mock_visuals, 
    mock_enrich, mock_calculate, mock_engineer, mock_assemble, mock_da
):
    # Setup mocks
    mock_assemble.return_value = pd.DataFrame({'Customer ID': ['1'], 'Industry': ['Tech']})
    mock_validate.return_value = pd.DataFrame({'Customer ID': ['1'], 'Industry': ['Tech']})
    
    mock_engineer.return_value = pd.DataFrame({
        'Customer ID': ['1'], 
        'ICP_score': [80], 
        'ICP_grade': ['B'],
        'adoption_score': [0.5],
        'relationship_score': [0.5],
        'vertical_score': [0.5],
        'ICP_score_raw': [80],
    })
    
    mock_calculate.return_value = pd.DataFrame({
        'Customer ID': ['1'], 
        'ICP_score': [70], 
        'ICP_grade': ['C'],
        'adoption_score': [0.4],
        'relationship_score': [0.4],
        'vertical_score': [0.4],
        'ICP_score_raw': [70],
        'Hardware_ICP_Score': [80], # Passed through
    })

    mock_enrich.return_value = pd.DataFrame({
        'Customer ID': ['1'], 
        'Hardware_ICP_Score': [80],
        'CRE_ICP_Score': [70]
    })
    
    mock_da.get_engine.return_value = MagicMock()
    mock_da.get_sales_detail_since_2022.return_value = pd.DataFrame()
    
    # Set CLI opts
    CLI_OPTS['division'] = 'hardware'
    CLI_OPTS['skip_neighbors'] = False # Enable neighbors
    CLI_OPTS['skip_visuals'] = True
    CLI_OPTS['out_path'] = 'test_output.csv'
    
    # Mock neighbors return
    mock_neighbors.return_value = pd.DataFrame({'source': ['1'], 'target': ['2']})
    
    # Run main
    main()
    
    # Verify neighbors called
    mock_neighbors.assert_called_once()

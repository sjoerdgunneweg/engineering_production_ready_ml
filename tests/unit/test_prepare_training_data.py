import os
import tempfile
import pandas as pd

from scripts.prepare_training_data import _combine_raw_tac_files, _add_tac_reading_to_windowed_data, _calculate_window_features

def test_combine_raw_tac_files():
    expected = pd.DataFrame({
        'pid': ['AB1234', 'AB1234', 'CD5678', 'CD5678'],
        'timestamp': pd.to_datetime([1, 2, 1, 2], unit='s'),
        'TAC_Reading': [10, 20, 30, 40]
    })
    

    with tempfile.TemporaryDirectory() as temp_dir:
        tac_file_1 = os.path.join(temp_dir, 'AB1234_clean_TAC.csv')
        tac_file_2 = os.path.join(temp_dir, 'CD5678_clean_TAC.csv')

        pd.DataFrame({
            'pid': ['A', 'A'],
            'timestamp': [1, 2],
            'TAC_Reading': [10, 20]
        }).to_csv(tac_file_1, index=False)

        pd.DataFrame({
            'pid': ['B', 'B'],
            'timestamp': [1, 2],
            'TAC_Reading': [30, 40]
        }).to_csv(tac_file_2, index=False)

        combined_df = _combine_raw_tac_files(temp_dir) 

    pd.testing.assert_frame_equal(combined_df.reset_index(drop=True), expected.reset_index(drop=True), check_like=True)

def test_combine_raw_tac_files_file_not_found():
    with tempfile.TemporaryDirectory() as temp_dir:
        combined_df = _combine_raw_tac_files(temp_dir) 
        assert combined_df.empty

def test_combine_raw_tac_files_invalid_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_file = os.path.join(temp_dir, 'invalid_clean_TAC.csv')
        with open(invalid_file, 'w') as f:
            f.write("not a valid csv")

        combined_df = _combine_raw_tac_files(temp_dir) 
        assert combined_df.empty

def test_combine_raw_tac_files_file_reading_error():
    with tempfile.TemporaryDirectory() as temp_dir:
        error_file = os.path.join(temp_dir, 'error_clean_TAC.csv')
        with open(error_file, 'w') as f:
            f.write("pid,timestamp,TAC_Reading\n1,2,three")  # 'three' will cause an error

        combined_df = _combine_raw_tac_files(temp_dir) 
        assert combined_df.empty


def test_calculate_window_features():
    acceleration_data = pd.DataFrame({
        'pid': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': [0, 5000, 10000, 15000, 20000, 25000],
        'x': [1, 2, 3, 4, 5, 6],
        'y': [2, 3, 4, 5, 6, 7],
        'z': [3, 4, 5, 6, 7, 8]
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        acceleration_data_path = os.path.join(temp_dir, 'temp_acceleration.csv') 
        acceleration_data.to_csv(acceleration_data_path, index=False)
        
        features_by_pid = _calculate_window_features(acceleration_data_path)

        expected_features = pd.DataFrame({
            'x': [1.5, 3.5, 5.5],
            'y': [2.5, 4.5, 6.5],
            'z': [3.5, 5.5, 7.5],
            'pid': ['A', 'A', 'A']
        }, index=[0.0, 10.0, 20.0]) # float index due to division in window calculation
        expected_features.index.name = 'window_start_time' 

        pd.testing.assert_frame_equal(features_by_pid['A'], expected_features, check_like=True)

def test_calculate_window_features_no_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        acceleration_data_path = os.path.join(temp_dir, 'empty_acceleration.csv') 
        pd.DataFrame(columns=['pid', 'time', 'x', 'y', 'z']).to_csv(acceleration_data_path, index=False)
        
        features_by_pid = _calculate_window_features(acceleration_data_path)

        assert features_by_pid == {}

def test_add_tac_reading_to_windowed_data():
    features_by_pid = {
        'A': pd.DataFrame({
            'x': [1.5, 3.5],
            'y': [2.5, 4.5],
            'z': [3.5, 5.5],
            'pid': ['A', 'A']
        }, index=[0.0, 10.0]) # float index due to division in window calculation
    }
    features_by_pid['A'].index.name = 'window_start_time'

    tac_data = pd.DataFrame({
        'pid': ['A', 'A'],
        'timestamp': [0, 10], 
        'TAC_Reading': [0.1, 0.02]
    })
    
    tac_data['timestamp'] = pd.to_datetime(tac_data['timestamp'], unit='s')

    combined_data = _add_tac_reading_to_windowed_data(features_by_pid, tac_data)

    expected_combined = pd.DataFrame({
        'x': [1.5, 3.5],
        'y': [2.5, 4.5],
        'z': [3.5, 5.5],
        'pid': ['A', 'A'],
        'TAC_Reading': [0.1, 0.02]
    }, index=[0.0, 10.0]) # float index due to division in window calculation
    expected_combined.index.name = 'window_start_time'

    pd.testing.assert_frame_equal(combined_data['A'], expected_combined, check_like=True)
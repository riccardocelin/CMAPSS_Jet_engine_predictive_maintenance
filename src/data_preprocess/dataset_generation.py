import json
import os
from pathlib import Path

import data_generation_fcn as data_gen
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'configs' / 'dataset_generation_config.local.json' # remember to create a new file copy from configs/dataset_generation_config.template.json


def load_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    required_fields = [
        'data_path',
        'dataset_training_patterns',
        'dataset_test_patterns',
        'dataset_test_rul_patterns',
        'dataset_processed_destination_path',
        'max_rul',
        'y_col_name',
        'sequence_len',
        'roll_mean_wd',
        'is_sequence_modeling',
        'dataset_version'
    ]
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise KeyError(f"Missing config fields: {', '.join(missing_fields)}")

    return config




def build_runtime_parameters(config):
    data_path = config['data_path']
    dataset_training_patterns = config['dataset_training_patterns']
    dataset_test_patterns = config['dataset_test_patterns']
    dataset_test_rul_patterns = config['dataset_test_rul_patterns']
    max_rul = config['max_rul']
    y_col_name = config['y_col_name']
    sequence_len = config['sequence_len']
    roll_mean_wd = config['roll_mean_wd']
    is_sequence_modeling = config['is_sequence_modeling']
    dataset_version = config['dataset_version']

    dataset_processed_destination_path = config['dataset_processed_destination_path']
    if is_sequence_modeling:
        dataset_processed_destination_path += '/sequence/' + dataset_version
    else:
        dataset_processed_destination_path += '/tabular/' + dataset_version

    prj_folder = os.path.abspath(os.curdir)
    dataset_processed_destination_path = prj_folder + '/' + dataset_processed_destination_path
    os.makedirs(dataset_processed_destination_path, exist_ok=True)

    return (
        data_path,
        dataset_training_patterns,
        dataset_test_patterns,
        dataset_test_rul_patterns,
        dataset_processed_destination_path,
        max_rul,
        y_col_name,
        sequence_len,
        roll_mean_wd,
        is_sequence_modeling,
        dataset_version
    )


def feature_engineering_pipeline_tabular(df, windows=[5]):

    df_eng = data_gen.sort_by_engine_and_cycle(df)

    SENSORS = [col for col in df_eng.columns if col not in ['engine_id', 'RUL']]

    # get the following rolling statistics for each sensor in fixed windows of cycles per engine:
    # - rolling mean
    # - trending difference between current value and rolling mean
    for w in windows:
        df_eng[[f'{sensor}_RollMean_{w}_Window' for sensor in SENSORS]] = (
            df_eng.groupby('engine_id')[SENSORS]
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df_eng[[f'{sensor}_inst_val_diff_wrt_RollMean_{w}_Window' for sensor in SENSORS]] = (
            df_eng[[sensor for sensor in SENSORS]].values - df_eng[[f'{sensor}_RollMean_{w}_Window' for sensor in SENSORS]].values
        )

    return data_gen.nan_management(df_eng)


def feature_engineering_pipeline_sequence(df, sequence_len=5, step_size=1, y_name='RUL'):
    X = []
    y = []

    df = data_gen.sort_by_engine_and_cycle(df)
    engine_ids = df['engine_id'].unique()
    
    for engine_id in engine_ids:
        engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
        features = engine_data.drop(columns=[y_name])
        y_col = engine_data[y_name]
        
        for start in range(0, len(engine_data) - sequence_len + 1, step_size):
            end = start + sequence_len
            X.append(features.values[start:end])
            y.append(y_col.values[start:end])

    return np.array(X), np.array(features.columns.values.tolist()), np.array(y), np.array([y_name])


def main(config_path=DEFAULT_CONFIG_PATH):

    config = load_config(config_path)
    (
        data_path,
        dataset_training_patterns,
        dataset_test_patterns,
        dataset_test_rul_patterns,
        dataset_processed_destination_path,
        max_rul,
        y_col_name,
        sequence_len,
        roll_mean_wd,
        is_sequence_modeling,
        dataset_version
    ) = build_runtime_parameters(config)

    # # training sets transformation
    for dataset in dataset_training_patterns:

        # load dataset
        df = data_gen.load_dataset(data_path, dataset, data_gen.column_names)

        # missing value management (replacement with previous value)
        df = data_gen.nan_management(df)

        # compute RUL feature
        df = data_gen.compute_training_rul(df, max_rul, y_col_name)

        # drop zero variance sensors
        df, cols_to_drop = data_gen.drop_zero_variance_features(df)

        # feature engineering step
        if is_sequence_modeling: # saved as npz file

            X_npArr, X_names_npArr, y_npArr, y_name_npArr = feature_engineering_pipeline_sequence(df, sequence_len, step_size=1, y_name=y_col_name) # returns a np array
            file_fullname = dataset_processed_destination_path + '/' + dataset + '_X_y_' + dataset_version
            np.savez(file=file_fullname+'.npz', X=X_npArr, y=y_npArr, feature_names=X_names_npArr, y_name=y_name_npArr)
        
        else: # saved as separate csv files

            df = feature_engineering_pipeline_tabular(df, roll_mean_wd)
            df.loc[:, df.columns != y_col_name].to_csv(dataset_processed_destination_path + '/' + dataset + '_X_' + dataset_version + '.csv', encoding='utf-8', index=False)
            df[y_col_name].to_csv(dataset_processed_destination_path + '/' + dataset + '_y_' + dataset_version + '.csv', encoding='utf-8', index=False)

    # test sets transformation
    for dataset, rul_file in zip(dataset_test_patterns, dataset_test_rul_patterns):

        # load dataset
        df = data_gen.load_dataset(data_path, dataset, data_gen.column_names)

        # missing value management (replacement with previous value)
        df = data_gen.nan_management(df)

        # compute RUL feature
        rul_at_last_available_cycle = data_gen.retrive_rul_from_rul_file(data_path, rul_file)
        df = data_gen.compute_test_rul(df, rul_at_last_available_cycle, max_rul, y_col_name)

        # drop zero variance sensors (same cols as in training set)
        df.drop(cols_to_drop, axis=1, inplace=True)

        # feature engineering step
        if is_sequence_modeling: # saved as npz file

            X_npArr, X_names_npArr, y_npArr, y_name_npArr = feature_engineering_pipeline_sequence(df, sequence_len, step_size=1, y_name=y_col_name) # returns a np array
            file_fullname = dataset_processed_destination_path + '/' + dataset + '_X_y_' + dataset_version
            np.savez(file=file_fullname+'.npz', X=X_npArr, y=y_npArr, feature_names=X_names_npArr, y_name=y_name_npArr)
        
        else: # saved as separate csv files

            df = feature_engineering_pipeline_tabular(df, roll_mean_wd)
            df.loc[:, df.columns != y_col_name].to_csv(dataset_processed_destination_path + '/' + dataset + '_X_' + dataset_version + '.csv', encoding='utf-8', index=False)
            df[y_col_name].to_csv(dataset_processed_destination_path + '/' + dataset + '_y_' + dataset_version + '.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
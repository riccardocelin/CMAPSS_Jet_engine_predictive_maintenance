import data_generation_fcn as data_gen
import numpy as np
import os


DATA_PATH = "data/CMAPSSData"

DATASET_TRAINING_PATTERNS = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']

DATASET_TEST_PATTERNS = ['test_FD001', 'test_FD002', 'test_FD003', 'test_FD004']
DATASET_TEST_RUL_PATTERNS = ['RUL_FD001', 'RUL_FD002', 'RUL_FD003', 'RUL_FD004']

DATASET_PROCESSED_DESTINATION_PATH = "data/processed"



#######################################################
MAX_RUL = 125 # clipping value (max RUL value)

Y_COL_NAME = 'RUL' # name for the y column

SEQUENCE_LEN = 10 # time length of the sequence ONLY FOR SEQUENCE PREPROCESSING (DL modeling)
ROLL_MEAN_WD = [5, 10, 15] # rolling mean windows ONLY FOR TABULAR PREPROCESSING (ML modeling)
is_sequence_modeling = True # True: enable dataset preprocessing for sequence modeling, False: enable dataset preprocessing for tabular modeling

dataset_version = 'v1'
#######################################################


if is_sequence_modeling:
    DATASET_PROCESSED_DESTINATION_PATH += '/sequence/' + dataset_version
else:
    DATASET_PROCESSED_DESTINATION_PATH += '/tabular/' + dataset_version

prj_folder = os.path.abspath(os.curdir)
DATASET_PROCESSED_DESTINATION_PATH = prj_folder + '/' + DATASET_PROCESSED_DESTINATION_PATH
os.makedirs(DATASET_PROCESSED_DESTINATION_PATH, exist_ok=True)


def feature_engineering_pipeline_tabular(df, windows=[5]):

    df_eng = df.copy()

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

    return df_eng.drop(columns='engine_id')


def feature_engineering_pipeline_sequence(df, sequence_len=5, step_size=1, y_name='RUL'):
    X = []
    y = []
    engine_ids = df['engine_id'].unique()
    
    for engine_id in engine_ids:
        engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
        features = engine_data.drop(columns=['engine_id', y_name])
        y_col = engine_data[y_name]
        
        for start in range(0, len(engine_data) - sequence_len + 1, step_size):
            end = start + sequence_len
            X.append(features.values[start:end])
            y.append(y_col.values[start:end])
    
    return np.array(X), np.array(features.columns.values.tolist()), np.array(y), np.array([y_name])


def main():

    # # training sets transformation
    for dataset in DATASET_TRAINING_PATTERNS:

        # load dataset
        df = data_gen.load_dataset(DATA_PATH, dataset, data_gen.column_names)

        # missing value management (replacement with previous value)
        df = data_gen.nan_management(df)

        # compute RUL feature
        df = data_gen.compute_training_rul(df, MAX_RUL, Y_COL_NAME)

        # drop zero variance sensors
        df, cols_to_drop = data_gen.drop_zero_variance_features(df)

        # feature engineering step
        if is_sequence_modeling: # saved as npz file

            X_npArr, X_names_npArr, y_npArr, y_name_npArr = feature_engineering_pipeline_sequence(df, SEQUENCE_LEN, step_size=1, y_name=Y_COL_NAME) # returns a np array
            file_fullname = DATASET_PROCESSED_DESTINATION_PATH + '/' + dataset + '_X_y_' + dataset_version
            np.savez(file=file_fullname+'.npz', X=X_npArr, y=y_npArr, feature_names=X_names_npArr, y_name=y_name_npArr)
        
        else: # saved as separate csv files

            df = feature_engineering_pipeline_tabular(df, ROLL_MEAN_WD)
            df.loc[:, df.columns != Y_COL_NAME].to_csv(DATASET_PROCESSED_DESTINATION_PATH + '/' + dataset + '_X_' + dataset_version, encoding='utf-8', index=False)
            df[Y_COL_NAME].to_csv(DATASET_PROCESSED_DESTINATION_PATH + '/' + dataset + '_y_' + dataset_version, encoding='utf-8', index=False)

    # test sets transformation
    for dataset, rul_file in zip(DATASET_TEST_PATTERNS, DATASET_TEST_RUL_PATTERNS):

        # load dataset
        df = data_gen.load_dataset(DATA_PATH, dataset, data_gen.column_names)

        # missing value management (replacement with previous value)
        df = data_gen.nan_management(df)

        # compute RUL feature
        rul_at_last_available_cycle = data_gen.retrive_rul_from_rul_file(DATA_PATH, rul_file)
        df = data_gen.compute_test_rul(df, rul_at_last_available_cycle, MAX_RUL, Y_COL_NAME)

        # drop zero variance sensors (same cols as in training set)
        df.drop(cols_to_drop, axis=1, inplace=True)

        # feature engineering step
        if is_sequence_modeling: # saved as npz file

            X_npArr, X_names_npArr, y_npArr, y_name_npArr = feature_engineering_pipeline_sequence(df, SEQUENCE_LEN, step_size=1, y_name=Y_COL_NAME) # returns a np array
            file_fullname = DATASET_PROCESSED_DESTINATION_PATH + '/' + dataset + '_X_y_' + dataset_version
            np.savez(file=file_fullname+'.npz', X=X_npArr, y=y_npArr, feature_names=X_names_npArr, y_name=y_name_npArr)
        
        else: # saved as separate csv files

            df = feature_engineering_pipeline_tabular(df, ROLL_MEAN_WD)
            df.loc[:, df.columns != Y_COL_NAME].to_csv(DATASET_PROCESSED_DESTINATION_PATH + '/' + dataset + '_X_' + dataset_version, encoding='utf-8', index=False)
            df[Y_COL_NAME].to_csv(DATASET_PROCESSED_DESTINATION_PATH + '/' + dataset + '_y_' + dataset_version, encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
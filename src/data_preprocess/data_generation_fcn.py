import pandas as pd

column_names = ['engine_id', 'cycle', 'altitude', 'mach_number', 'throttle_angle', 'fan_inlet_temp',
              'LPC_outlet_temp', 'HPC_outlet_temp', 'LPT_outlet_temp', 'fan_inlet_press', 'bypass_press',
              'HPC_outlet_press', 'fan_speed', 'core_speed', 'engine_pressure_ratio', 'HPC_static_outlet_press',
              'fuel_flow_ratio', 'corrected_fan_speed', 'corrected_core_speed', 'bypass_ratio', 'burner_fuel_air_ratio',
              'bleed_enthalpy', 'demanded_fan_speed', 'demanded_corrected_fan_speed', 'HPT_coolant_bleed', 'LPT_coolant_bleed']


def load_dataset(data_path, dataset, column_names):

    df = pd.read_csv(f'{data_path}/{dataset}.txt', sep='\s+', header=None, index_col=False)
    df.columns = list(column_names)
        
    return df


def compute_training_rul(df, max_rul=150, y_name='RUL'):
    
    rul_data = df.groupby('engine_id')['cycle'].max().reset_index()
    rul_data.columns = ['engine_id', 'max_cycle']

    df = df.merge(rul_data, on='engine_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']

    df[y_name] = df[y_name].clip(upper=max_rul)

    df.drop('max_cycle', axis=1, inplace=True)

    return df

  
def compute_test_rul(df, rul_last, max_rul=150, y_name='RUL'):

    rul_last_df = pd.DataFrame({
        "engine_id": df["engine_id"].unique(),
        "rul_last": rul_last[0]
    })

    last_cycle = df.groupby('engine_id')['cycle'].transform('max')

    df = df.merge(rul_last_df, on='engine_id', how='left')

    df[y_name] = (last_cycle - df['cycle']) + df['rul_last']
    df[y_name] = df[y_name].clip(upper=max_rul)

    df.drop('rul_last', axis=1, inplace=True)

    return df


def nan_management(df):
    df.ffill(inplace=True)
    return df


def drop_zero_variance_features(df):
    # drop non informative sensors (with constant values)
    DROP_SENSORS = []

    for feature in df.columns:
        try:
            if df[feature].min()==df[feature].max():
                DROP_SENSORS.append(feature)
        except:
            pass

    df.drop(DROP_SENSORS,axis=1,inplace=True)

    return df, DROP_SENSORS


def retrive_rul_from_rul_file(data_path, rul_file):

    df = pd.read_csv(f'{data_path}/{rul_file}.txt', sep='\s+', header=None, index_col=False)
    return df


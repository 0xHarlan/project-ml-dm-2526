import pandas as pd
import numpy as np

# A simple feature engineering using blocks means, stds, simple ratios on S1 sensors & humidity and an attempt of removing
# humidity bias using the function described under
def feature_engineering(df: pd.DataFrame):
    df = df.copy()

    block1 = ['M12', 'M13', 'M14', 'M15']
    block2 = ['M4', 'M5', 'M6', 'M7']

    df = remove_humidity_bias(df, sensors=block1+block2)
    
    block1_clean = [f"{c}_clean" for c in block1]
    block2_clean = [f"{c}_clean" for c in block2]

    df["block1_mean"] = df[block1_clean].mean(axis=1)
    df["block2_mean"] = df[block2_clean].mean(axis=1)
    df["block1_std"] = df[block1_clean].std(axis=1)
    df["block2_std"] = df[block2_clean].std(axis=1)

    eps = 1e-6
        
    for col in (block1_clean+block2_clean):
        df[f'{col}_ratio_S1'] = df[col] / (df['S1'] + eps)
        df[f'{col}_ratio_Hum'] = df[col] / (df['Humidity'] + eps)
        df[f'{col}_ratio_R'] = df[col] / (df['R'] + eps)

    return df

# An attempt of removing the humidity bias on the sensors, using k-Köhler simplified formula described in:
# Section 3.3: https://pure-oai.bham.ac.uk/ws/portalfiles/portal/48965889/Crilley_et_al_Evaluation_low_cost_optical_Atmospheric_Measuring_Techniques.pdf
# Section 2.1: https://acp.copernicus.org/articles/7/1961/2007/acp-7-1961-2007.pdf
def remove_humidity_bias(df: pd.DataFrame, sensors = ['M4', 'M5', 'M6', 'M7', 'M12', 'M13', 'M14', 'M15']):
    df = df.copy()
    # We clip to 0.99 to avoid dividing by zero (max value of humidity in x_train: 0.958254588759158)
    rh_safe = df['Humidity'].clip(upper=0.99)
    k = 0.35
    threshold = 0.7

    correction_factor = pd.Series(1.0, index=df.index)

    # We find the columns where the humidity is higher than the arbitrary threshold
    mask_high_rh = rh_safe > threshold

    # k-Köhler formula
    # C = 1 + k * (RH / (1 - RH))
    correction_factor[mask_high_rh] = 1 + k * (rh_safe[mask_high_rh] / (1 - rh_safe[mask_high_rh]))

    for sensor in sensors:
        df[f"{sensor}_clean"] = df[sensor] / correction_factor

    return df

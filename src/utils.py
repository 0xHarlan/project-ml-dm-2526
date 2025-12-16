import pandas as pd

# Loads datasets, from csv to DataFrames ( for future feature engineering/data manipulation )
def read_datasets(x_train_fname, y_train_fname, x_test_fname) -> pd.DataFrame:    
    x_train = pd.read_csv(x_train_fname)
    y_train = pd.read_csv(y_train_fname)
    x_test  = pd.read_csv(x_test_fname)

    # Just in case
    x_train = x_train.sort_values("ID")
    y_train = y_train.sort_values("ID")

    return x_train, y_train, x_test

# Splits x_train, y_train into x_train, y_train, x_test, y_test for validation -> Obsolete
def split(x_train, y_train, size = 0.2):
    split_index = int(len(x_train) * (1 - size))
    
    # First part
    x_train_split = x_train.iloc[:split_index]
    y_train_split = y_train.iloc[:split_index]
    
    # Second part
    x_test_split = x_train.iloc[split_index:]
    y_test_split = y_train.iloc[split_index:]
    
    # Sklearn convention train/test
    return x_train_split, x_test_split, y_train_split, y_test_split

# Splits on a "domain" in this case, humidity, instead of ID. So that the model doesnt overfit
def humidity_split(x_train, y_train, size = 0.2):
    threshold = x_train['Humidity'].quantile(size) # We find the value for humidity thats splits the data into n*(1-size)/n*size

    # We create the masks, on default size parameter, train datasets becomes low to [threshold] humidity, test becomes high humidity
    mask_train = x_train['Humidity'] <= threshold
    mask_test = x_train['Humidity'] > threshold

    x_train_split = x_train[mask_train]
    y_train_split = y_train[mask_train]

    x_test_split = x_train[mask_test]
    y_test_split = y_train[mask_test]

    return x_train_split, x_test_split, y_train_split, y_test_split

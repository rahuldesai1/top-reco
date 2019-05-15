import pandas as pd
import numpy as np
import sys

def load_data(data_path, delimiter=','):
    data = pd.read_csv(data_path, delimiter=delimiter)    
    try:
        data = data.drop('Unnamed: 0', axis=1).drop('weight', axis=1).drop('rndm', axis=1)
    except:
        pass       
    return data

def process(data_path, output_path, delimiter=",", save=False):
    df = pd.read_csv(data_path, delimiter=delimiter)

    #dataframe preprocessing
    y = final_df['label']
    X = final_df.drop('label', axis=1)

    #data normalization
    normalization_data = pd.DataFrame(columns=["column_name", "slope_upper", "slope_lower", "threshold", "minimum"])

    def normalize(col):
        global normalization_data
        print(col.name)
        threshold = col.quantile(0.9)
        mini = col.min()
        slopeUpper = (1 - 0.9) / (col.max() - threshold)
        slopeLower = (0.6 - 0) / (threshold - mini)
        temp = {"column_name": col.name, "slope_upper": slopeUpper, "slope_lower": slopeLower, "threshold": threshold, "minimum": mini}
        normalization_data = normalization_data.append(temp, ignore_index=True)
        def norm_helper(row):
            if row > threshold:
                return 0.9 + slopeUpper * (row - threshold)
            else:
                return 0 + slopeLower * (row - mini)
        return col.apply(norm_helper)

    #normalize the data
    print("[PROCESSING] Normalzing data")
    X_norm = X.apply(normalize, axis=0)

    normalized = pd.concat([y, X_norm], axis=1)
    if save:
        normalized.to_csv(output_path + '/norm_results.csv'.format(frac), sep=',')
        normalization_data.to_csv(output_path + '/scalar_data.csv', sep=',')
        print("[INFO] Saving scalar data and normalized data to {0}".format(output_path))
    else:
        normalization_data.to_csv(output_path + '/scalar_data.csv', sep=',')
        print("[INFO] Saving scalar data to {0}".format(output_path))

    return normalized

def normalize(data_path, scalar_data_dir, delimiter=','):
    print("[PROCESSING] Loading data")
    normalizer = pd.read_csv(scalar_data_dir + "/scalar_data.csv", delimiter=delimiter)
    data = pd.read_csv(data_path, delimiter=',')
    print("[INFO] Done loading data")

    def norm(col):
        norm_col = normalizer.loc[normalizer["column_name"] == str(col.name)]
        threshold = norm_col["threshold"].values[0]
        mini = norm_col["minimum"].values[0]
        slopeUpper = norm_col["slope_upper"].values[0]
        slopeLower = norm_col["slope_lower"].values[0]
        def norm_helper(row):
            if row > threshold:
                return 0.9 + slopeUpper * (row - threshold)
            else:
                return 0 + slopeLower * (row - mini)
        return col.apply(norm_helper)

    y = data['label']
    data = data.drop('label', axis=1)

    print("[PROCESSING] Normalzing data")
    X_norm = data.apply(norm, axis=0)
    normalized = pd.concat([y, X_norm], axis=1)

    return normalized

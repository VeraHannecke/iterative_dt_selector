import os
import pandas as pd


def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(file_path, header=None)
    elif ext == '.csv':

        return pd.read_csv(file_path, header=None)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Accepted formats: .xlsx, .xls, .csv")


def prepare_dataset(file_path, threshold):
    raw = load_data(file_path)

    features = raw.iloc[2:, 0].tolist()
    sample_ids = raw.iloc[0, 1:].astype(str).tolist()
    regression = raw.iloc[1, 1:].values.astype(float)
    
    matrix = raw.iloc[2:, 1:].values.astype(float)

    # needs to be samples x genes for sklearn
    X = pd.DataFrame(matrix.T, columns=features, index=sample_ids)
    y = (regression <= threshold).astype(int)  # convert to binary classes

    return X, y

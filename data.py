import ast
from functools import reduce
import json
import os
import pandas as pd
import sklearn as skl

def getdata(path):
    data = []

    for file in os.listdir(path):
        if 'grouped' in file.lower():
            filepath = os.path.join(path, file)
            with open (filepath, 'r') as f:
                read = json.load(f)
                data.extend(read)

    return data

def expand_ranking_column(df, col_name):
    # Safely convert strings to lists (skip if already a list)
    df.loc[:, col_name] = df[col_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Expand list into separate columns
    new_cols = pd.DataFrame(df[col_name].tolist(), index=df.index)
    new_cols.columns = [f"ranking_{i}" for i in range(new_cols.shape[1])]
    
    # Drop original column and concatenate new columns
    df = pd.concat([df.drop(columns=[col_name]), new_cols], axis=1)
    return df


def builddata(type):
    path_dataset = 'data/dataset_detection'
    helpsteer2_data = pd.DataFrame()
    helpsteer3_data = pd.DataFrame()
    antique_data = pd.DataFrame()
    neurips_data = pd.DataFrame()

    for subfolder in os.listdir(path_dataset):
        subfolder_path = os.path.join(path_dataset, subfolder)
        if os.path.isdir(subfolder_path) and type in subfolder.lower():
            data = getdata(subfolder_path)
            df = pd.DataFrame(data)
            if 'helpsteer2' in subfolder.lower():
                temp = df[['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity', 'label']]
                helpsteer2_data = pd.concat([helpsteer2_data, temp], ignore_index=True)
            elif 'helpsteer3' in subfolder.lower():
                temp = df[['score', 'label']]
                helpsteer3_data = pd.concat([helpsteer3_data, temp], ignore_index=True)
            elif 'antique' in subfolder.lower():
                temp = df[['ranking', 'label']]
                temp = expand_ranking_column(temp, 'ranking')
                antique_data = pd.concat([antique_data, temp], ignore_index=True)
            elif 'neurips' in subfolder.lower():
                temp = df[['rating', 'confidence', 'soundness', 'presentation', 'contribution', 'label']]
                neurips_data = pd.concat([neurips_data, temp], ignore_index=True)
    
    dfs = [helpsteer2_data, helpsteer3_data, antique_data, neurips_data]

    return dfs


def getfeature_data(type):
    path = 'data/dataset_detection/features'

    helpsteer2_df = pd.DataFrame()
    helpsteer3_df = pd.DataFrame()
    antique_df = pd.DataFrame()
    neurips_df = pd.DataFrame()
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if (os.path.isdir(subfolder_path)):
            if 'linguistic' in subfolder.lower():
                for file in os.listdir(subfolder_path):
                    filepath = os.path.join(subfolder, file)
                    if 'helpsteer2' in file.lower() and type in file.lower():
                        data = pd.read_csv(filepath)
                        helpsteer2_df = pd.concat([helpsteer2_df, data], ignore_index=True)
                    elif 'helpsteer3' in file.lower() and type in file.lower():
                        data = pd.read_csv(filepath)

            
        
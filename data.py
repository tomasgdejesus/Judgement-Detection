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

    helpsteer2_data_comp = pd.DataFrame()
    helpsteer3_data_comp = pd.DataFrame()
    antique_data_comp = pd.DataFrame()
    neurips_data_comp = pd.DataFrame()

    for subfolder in os.listdir(path_dataset):
        subfolder_path = os.path.join(path_dataset, subfolder)
        if os.path.isdir(subfolder_path) and type in subfolder.lower():
            data = getdata(subfolder_path)
            df = pd.DataFrame(data)
            if 'helpsteer2' in subfolder.lower():
                temp = df[['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity', 'label']]
                helpsteer2_data = pd.concat([helpsteer2_data, temp], ignore_index=True)
                helpsteer2_data_comp = pd.concat([helpsteer2_data_comp, df], ignore_index=True)
            elif 'helpsteer3' in subfolder.lower():
                temp = df[['score', 'label']]
                helpsteer3_data = pd.concat([helpsteer3_data, temp], ignore_index=True)
                helpsteer3_data_comp = pd.concat([helpsteer3_data_comp, df], ignore_index=True)
            elif 'antique' in subfolder.lower():
                temp = df[['ranking', 'label']]
                temp = expand_ranking_column(temp, 'ranking')
                antique_data = pd.concat([antique_data, temp], ignore_index=True)
                antique_data_comp = pd.concat([antique_data_comp, df], ignore_index=True)
            elif 'neurips' in subfolder.lower():
                temp = df[['rating', 'confidence', 'soundness', 'presentation', 'contribution', 'label']]
                neurips_data = pd.concat([neurips_data, temp], ignore_index=True)
                neurips_data_comp = pd.concat([neurips_data_comp, df], ignore_index=True)
    
    dfs = [helpsteer2_data, helpsteer3_data, antique_data, neurips_data]
    comp_dfs = [helpsteer2_data_comp, helpsteer3_data_comp, antique_data_comp, neurips_data_comp]

    return dfs, comp_dfs


def getfeature_data(type):
    path = 'data/features'

    helpsteer2_df = pd.DataFrame()
    helpsteer3_df = pd.DataFrame()
    antique_df = pd.DataFrame()
    neurips_df = pd.DataFrame()
    helpsteer2_df_llm = pd.DataFrame()
    helpsteer3_df_llm = pd.DataFrame()
    antique_df_llm = pd.DataFrame()
    neurips_df_llm = pd.DataFrame()
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if (os.path.isdir(subfolder_path)):
            if 'linguistic' in subfolder.lower():
                for file in os.listdir(subfolder_path):
                    filepath = os.path.join(subfolder_path, file)
                    if 'helpsteer2' in file.lower() and type in file.lower():
                        data = pd.read_csv(filepath)
                        helpsteer2_df = pd.concat([helpsteer2_df, data], ignore_index=True)
                    elif 'helpsteer3' in file.lower() and type in file.lower():
                        data = pd.read_csv(filepath)
                        helpsteer3_df = pd.concat([helpsteer3_df, data], ignore_index=True)
                    elif 'antique' in file.lower() and type in file.lower():
                        data = pd.read_csv(filepath)
                        data["docs"] = data[["response1", "response2", "response3"]].apply(lambda row: tuple([x for x in row if pd.notna(x)]), axis=1)
                        data = data.drop(columns=["response1", "response2", "response3"], axis=1)
                        antique_df = pd.concat([antique_df, data], ignore_index=True)
                    elif 'neurips' in file.lower() and type in file.lower():
                        data = pd.read_csv(filepath)
                        neurips_df = pd.concat([neurips_df, data], ignore_index=True)
            if 'llm' in subfolder.lower():
                for file in os.listdir(subfolder_path):
                    filepath = os.path.join(subfolder_path, file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        llm_rows = []
                    if 'helpsteer2' in file.lower() and type in file.lower():
                        for x in data:
                            feats = x.get("llm_enhanced_feature")
                            if feats != None:
                                llm_rows.append({
                                    "style_score": feats.get("style score", 0),
                                    "format_score": feats.get("format score", 0),
                                    "wording_score": feats.get("wording score", 0),
                                    "helpfulness_score": feats.get("helpfulness score", 0),
                                    "correctness_score": feats.get("correctness score", 0),
                                    "coherence_score": feats.get("coherence score", 0),
                                    "complexity_score": feats.get("complexity score", 0),
                                    "verbosity_score": feats.get("verbosity score", 0),
                                    "prompt": x["prompt"],
                                    "response": x["response"]
                                })

                        df_llm = pd.DataFrame(llm_rows)
                        helpsteer2_df_llm = pd.concat([helpsteer2_df_llm, df_llm], ignore_index=True)
                    elif 'helpsteer3' in file.lower() and type in file.lower():
                        for x in data:
                            feats = x.get("llm_enhanced_feature_r1")
                            feats2 = x.get("llm_enhanced_feature_r2")
                            if feats != None and feats2 != None:
                                llm_rows.append({
                                    "style_score": feats.get("style score", 0),
                                    "format_score": feats.get("format score", 0),
                                    "wording_score": feats.get("wording score", 0),
                                    "helpfulness_score": feats.get("helpfulness score", 0),
                                    "correctness_score": feats.get("correctness score", 0),
                                    "coherence_score": feats.get("coherence score", 0),
                                    "complexity_score": feats.get("complexity score", 0),
                                    "verbosity_score": feats.get("verbosity score", 0),
                                    "style_score2": feats2.get("style score", 0),
                                    "format_score2": feats2.get("format score", 0),
                                    "wording_score2": feats2.get("wording score", 0),
                                    "helpfulness_score2": feats2.get("helpfulness score", 0),
                                    "correctness_score2": feats2.get("correctness score", 0),
                                    "coherence_score2": feats2.get("coherence score"),
                                    "complexity_score2": feats2.get("complexity score", 0),
                                    "verbosity_score2": feats2.get("verbosity score", 0),
                                    "response1": x["response1"],
                                    "response2": x["response2"]
                                })

                        df_llm = pd.DataFrame(llm_rows)
                        helpsteer3_df_llm = pd.concat([helpsteer3_df_llm, df_llm], ignore_index=True)
                    elif 'antique' in file.lower() and type in file.lower():
                        for x in data:
                            feats = x.get("llm_enhanced_feature")
                            if feats != None:
                                llm_rows.append({
                                    "r1_score": feats["Response1 Score"],
                                    "r2_score": feats["Response2 Score"],
                                    "r3_score": feats["Response3 Score"],
                                    "query": x["query"],
                                    "docs": tuple(x["docs"])
                                })

                        df_llm = pd.DataFrame(llm_rows)
                        antique_df_llm = pd.concat([antique_df_llm, df_llm], ignore_index=True)
                    elif 'neurips' in file.lower() and type in file.lower():
                        for x in data:
                            feats = x.get("llm_enhanced_feature")
                            if feats != None:
                                llm_rows.append({
                                    "style_score": feats.get("style score", 0),
                                    "format_score": feats.get("format score", 0),
                                    "wording_score": feats.get("wording score", 0),
                                    "rating_score": feats.get("rating score", 0),
                                    "confidence_score": feats.get("confidence score", 0),
                                    "soundness_score": feats.get("soundness score", 0),
                                    "presentation_score": feats.get("presentation score", 0),
                                    "contribution_score": feats.get("contribution score", 0),
                                    "content": tuple(x["content"])
                                })
                        
                        df_llm = pd.DataFrame(llm_rows)
                        neurips_df_llm = pd.concat([neurips_df_llm, df_llm], ignore_index=True)
    
    temp = helpsteer2_df.merge(helpsteer2_df_llm, on=["prompt", "response"])
    helpsteer2_df = temp

    temp = helpsteer3_df.merge(helpsteer3_df_llm, on=["response1", "response2"])
    helpsteer3_df = temp

    temp = antique_df.merge(antique_df_llm, on=["query", "docs"])
    antique_df = temp

    temp = neurips_df.merge(neurips_df_llm, on="content")
    neurips_df = temp

    dfs = [helpsteer2_df, helpsteer3_df, antique_df, neurips_df]
    return dfs

def getcombine_features(type):
    df_all = []
    feature_dfs = getfeature_data(type)
    raw_dfs, df_comp = builddata(type)
    
    # helpsteer2
    temp = df_comp[0].merge(feature_dfs[0], on=["prompt", "response"])
    temp = temp.drop(columns=["prompt", "response"], axis=1)
    df_all.append(temp)

    # helpsteer3
    temp = df_comp[1].merge(feature_dfs[1], on=["response1", "response2"])
    temp = temp.drop(columns=["response1", "response2"], axis=1)
    df_all.append(temp)

    # antique
    df_comp[2]["docs"] = df_comp[2]["docs"].apply(tuple)
    temp = df_comp[2].merge(feature_dfs[2], on=["query", "docs"])
    temp = temp.drop(columns=["query", "docs"], axis=1)
    df_all.append(temp)

    # neurips
    df_comp[3]["content"] = df_comp[3]["content"].apply(tuple)
    temp = df_comp[3].merge(feature_dfs[3], on="content")
    temp = temp.drop(columns=["content"], axis=1)
    df_all.append(temp)

    return df_all


            
        
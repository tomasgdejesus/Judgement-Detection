import data as data
import mlmodel as model


if __name__ == "__main__":
    print("hi")

    # Build the dataset wiht only judgement decisions
    train_data, train_data_comp = data.builddata('train')
    test_data, test_data_comp = data.builddata('test')


    train_data[0].to_csv('helpsteer2.csv', index=False)
    train_data[1].to_csv('helpsteer3.csv', index=False)
    train_data[2].to_csv('antique.csv', index=False)
    train_data[3].to_csv('neurips.csv', index=False)
    
    # Base classifier
    model.getreport(train_data[0], test_data[0], 'helpsteer2')
    model.getreport(train_data[1], test_data[1], 'helpsteer3')
    model.getreport(train_data[2], test_data[2], 'antique')
    model.getreport(train_data[3], test_data[3], 'neurips')

    # augmented features (judgement + llm + linguistic)
    train_data = data.getcombine_features('train')
    test_data = data.getcombine_features('test')

    # Check if any null values are there in dataframes
    
    # helpsteer2 dataset
    num_rows_with_nan = train_data[0].isnull().any(axis=1).sum()
    print(f"Helpsteer2 dataset null values: {num_rows_with_nan}")
    if num_rows_with_nan > 0:
        train_data[0] = train_data[0].dropna()
        print(f"Num of records after removing null values for Helpsteer2 dataset: {train_data[0].shape[0]}")
    
    # helpsteer3 dataset
    num_rows_with_nan = train_data[1].isnull().any(axis=1).sum()
    print(f"Helpsteer3 dataset null values: {num_rows_with_nan}")
    if num_rows_with_nan > 0:
        train_data[1] = train_data[1].dropna()
        print(f"Num of records after removing null values for Helpsteer3 dataset: {train_data[1].shape[0]}")
    
    # antique dataset
    num_rows_with_nan = train_data[2].isnull().any(axis=1).sum()
    print(f"Antique dataset null values: {num_rows_with_nan}")
    if num_rows_with_nan > 0:
        train_data[2] = train_data[2].dropna()
        print(f"Num of records after removing null values for Antique dataset: {train_data[2].shape[0]}")
    
    # neurips dataset
    num_rows_with_nan = train_data[3].isnull().any(axis=1).sum()
    print(f"Neurips dataset null values: {num_rows_with_nan}")
    if num_rows_with_nan > 0:
        train_data[3] = train_data[3].dropna()
        print(f"Num of records after removing null values for Neurips dataset: {train_data[3].shape[0]}")
    
    # Generate csv files
    train_data[0].to_csv('helpsteer2_combine.csv', index=False)
    train_data[1].to_csv('helpsteer3_combine.csv', index=False)
    train_data[2].to_csv('antique_combine.csv', index=False)
    train_data[3].to_csv('neurips_combine.csv', index=False)

    # Base classifier
    model.getreport(train_data[0], test_data[0], 'helpsteer2')
    model.getreport(train_data[1], test_data[1], 'helpsteer3')
    model.getreport(train_data[2], test_data[2], 'antique')
    model.getreport(train_data[3], test_data[3], 'neurips')
    

    
                
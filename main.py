import csv
import data as data
import mlmodel as model
import plot as pl


if __name__ == "__main__":
    print("hi")

    # CREATE CSV WITH HEADER
    with open('metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Model', 'Dataset', 'Accuracy', 'F1'])
    
    # Build the dataset with only judgement decisions
    train_data, train_data_comp = data.getrawdata('train', True, 1)
    test_data, test_data_comp = data.getrawdata('test', True, 1)

    # For debug purpose
    # Generate csv files 
    data.write_to_csv(train_data)
    
    # Base classifier
    m_helpsteer2lr, m_helpsteer2rf, m_helpsteer2lg, scaler_helpsteer2 = model.getreport(train_data[0], test_data[0], 'helpsteer2', "Judgement Features")
    m_helpsteer3lr, m_helpsteer3rf, m_helpsteer3lg, scaler_helpsteer3 = model.getreport(train_data[1], test_data[1], 'helpsteer3', "Judgement Features")
    m_antiquelr, m_antiquerf, m_antiquelg, scaler_antique = model.getreport(train_data[2], test_data[2], 'antique', "Judgement Features")
    m_neuripslr, m_neuripsrf, m_neuripslg, scaler_neurips = model.getreport(train_data[3], test_data[3], 'neurips', "Judgement Features")
    modelslr = [m_helpsteer2lr, m_helpsteer3lr, m_antiquelr, m_neuripslr]
    modelsrf = [m_helpsteer2rf, m_helpsteer3rf, m_antiquerf, m_neuripsrf]
    modelslg = [m_helpsteer2lg, m_helpsteer3lg, m_antiquelg, m_neuripslg]
    scaler = [scaler_helpsteer2, scaler_helpsteer3, scaler_antique, scaler_neurips]

    # Build the dataset with augmented features (judgement + llm + linguistic)
    train_data = data.getcombine_features('train')
    test_data = data.getcombine_features('test')


    # For debug purpose
    # Generate csv files 
    data.write_to_csv(train_data, 'Combine')

    # Base classifier
    model.getreport(train_data[0], test_data[0], 'helpsteer2', "Augmented features")
    model.getreport(train_data[1], test_data[1], 'helpsteer3', "Augmented features")
    model.getreport(train_data[2], test_data[2], 'antique', "Augmented features")
    model.getreport(train_data[3], test_data[3], 'neurips', "Augmented features")
    

    # Group Detection
    datasets = ['helpsteer2', 'helpsteer3', 'antique', 'neurips']
    # Store results
    results_lr_acc = {ds: [] for ds in datasets}
    results_rf_acc = {ds: [] for ds in datasets}
    results_lg_acc = {ds: [] for ds in datasets}
    results_lr_f1 = {ds: [] for ds in datasets}
    results_rf_f1 = {ds: [] for ds in datasets}
    results_lg_f1 = {ds: [] for ds in datasets}

    k = [1, 2, 4, 8 ,16]
    for i in range(len(k)):
        test_data, test_data_comp = data.getrawdata('test', False, k[i])
        metrics = model.predict(test_data_comp, modelslr, modelsrf, modelslg, scaler, f"GroupSize_{k[i]}")

        # Store accuracies for each dataset
        for i, dataset in enumerate(datasets):
            results_lr_acc[dataset].append(metrics[i][0])  # LR accuracy
            results_rf_acc[dataset].append(metrics[i][1])  # RF accuracy
            results_lg_acc[dataset].append(metrics[i][2])  # LR accuracy
            results_lr_f1[dataset].append(metrics[i][0])  # LR f1 score
            results_rf_f1[dataset].append(metrics[i][1])  # RF f1 score
            results_lg_f1[dataset].append(metrics[i][2])  # LR f1 score
    
    pl.plot_all(datasets, {
            "Logistic Regression": results_lr_acc,
            "Random Forest": results_rf_acc,
            "LightGBM": results_lg_acc
        }, {
            "Logistic Regression": results_lr_f1,
            "Random Forest": results_rf_f1,
            "LightGBM": results_lg_f1
        }, k)


    # train_combined = data.getcombine_features('train')
    # for i, name in enumerate(['helpsteer2', 'helpsteer3', 'antique', 'neurips']):
    #     print(f"{name}:")
    #     print(f"  Shape: {train_combined[i].shape}")
    #     print(f"  Features (excluding label): {train_combined[i].shape[1] - 1}")
                    
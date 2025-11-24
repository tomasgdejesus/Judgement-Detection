import csv
import data as data
import mlmodel as model
import plot as pl


def group_detection(models_lr, models_rf, scalar_base, isAugmentedFeature):
    datasets = ['helpsteer2', 'helpsteer3', 'antique', 'neurips']
    # Store results
    results_lr_acc = {ds: [] for ds in datasets}
    results_rf_acc = {ds: [] for ds in datasets}
    results_lr_f1 = {ds: [] for ds in datasets}
    results_rf_f1 = {ds: [] for ds in datasets}

    k = [1, 2, 4, 8 ,16]
    feature_data = data.getfeature_data('test')
    for i in range(len(k)):
        test_data, test_data_comp = data.getrawdata('test', False, k[i])
        metrics = model.predict(test_data_comp, models_lr, models_rf, scalar_base, f"GroupSize_{k[i]}_Aug:{isAugmentedFeature}", feature_data, isAugmentedFeature)

        # Store accuracies for each dataset
        for i, dataset in enumerate(datasets):
            results_lr_acc[dataset].append(metrics[i][0])  # LR accuracy
            results_rf_acc[dataset].append(metrics[i][1])  # RF accuracy
            results_lr_f1[dataset].append(metrics[i][0])  # LR f1 score
            results_rf_f1[dataset].append(metrics[i][1])  # RF f1 score
    
    pl.plot_all(datasets, {
            "Logistic Regression": results_lr_acc,
            "Random Forest": results_rf_acc
        }, {
            "Logistic Regression": results_lr_f1,
            "Random Forest": results_rf_f1
        }, k, isAugmentedFeature)


if __name__ == "__main__":
    print("hi")

    # CREATE CSV WITH HEADER
    with open('metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Model', 'Dataset', 'Accuracy', 'F1'])
    
    # Build the dataset with only judgement decisions
    train_data_base, train_data_comp = data.getrawdata('train', True, 1)
    test_data_base, test_data_comp = data.getrawdata('test', True, 1)

    # For debug purpose
    # Generate csv files 
    data.write_to_csv(train_data_base)
    
    # Base classifier
    m_helpsteer2lr, m_helpsteer2rf, scaler_helpsteer2 = model.getreport(train_data_base[0], test_data_base[0], 'helpsteer2', "Judgement Features")
    m_helpsteer3lr, m_helpsteer3rf, scaler_helpsteer3 = model.getreport(train_data_base[1], test_data_base[1], 'helpsteer3', "Judgement Features")
    m_antiquelr, m_antiquerf, scaler_antique = model.getreport(train_data_base[2], test_data_base[2], 'antique', "Judgement Features")
    m_neuripslr, m_neuripsrf, scaler_neurips = model.getreport(train_data_base[3], test_data_base[3], 'neurips', "Judgement Features")
    modelslr = [m_helpsteer2lr, m_helpsteer3lr, m_antiquelr, m_neuripslr]
    modelsrf = [m_helpsteer2rf, m_helpsteer3rf, m_antiquerf, m_neuripsrf]
    scaler_base = [scaler_helpsteer2, scaler_helpsteer3, scaler_antique, scaler_neurips]

    # Build the dataset with augmented features (judgement + llm + linguistic)
    train_data = data.getcombine_features('train')
    test_data = data.getcombine_features('test')


    # For debug purpose
    # Generate csv files 
    data.write_to_csv(train_data, 'Combine')

    # Base classifier with augmented features
    augm_helpsteer2lr, augm_helpsteer2rf, augscaler_helpsteer2 = model.getreport(train_data[0], test_data[0], 'helpsteer2', "Augmented features")
    augm_helpsteer3lr, augm_helpsteer3rf, augscaler_helpsteer3 = model.getreport(train_data[1], test_data[1], 'helpsteer3', "Augmented features")
    augm_antiquelr, augm_antiquerf, augscaler_antique = model.getreport(train_data[2], test_data[2], 'antique', "Augmented features")
    augm_neuripslr, augm_neuripsrf, augscaler_neurips = model.getreport(train_data[3], test_data[3], 'neurips', "Augmented features")
    
    augmodelslr = [augm_helpsteer2lr, augm_helpsteer3lr, augm_antiquelr, augm_neuripslr]
    augmodelsrf = [augm_helpsteer2rf, augm_helpsteer3rf, augm_antiquerf, augm_neuripsrf]
    augscaler = [augscaler_helpsteer2, augscaler_helpsteer3, augscaler_antique, augscaler_neurips]

    # Group Detection
    group_detection(modelslr, modelsrf, scaler_base, False)
    group_detection(augmodelslr, augmodelsrf, augscaler, True)
    
    # With Judgement Dimensions
    # Rating Scale Analysis (helpsteer2 and helpsteer3)
    run_metadata = {'type' : 'base classifier', 'rating_scale' : 'binary'}
    model.getreport(data.get_binarized_rating_scale(train_data_base[0], 'helpsteer2'), data.get_binarized_rating_scale(test_data_base[0], 'helpsteer2'), 'helpsteer2', "Ratingscale Analysis")
    model.getreport(data.get_binarized_rating_scale(train_data_base[1], 'helpsteer3'), data.get_binarized_rating_scale(test_data_base[1], 'helpsteer3'), 'helpsteer3', "Ratingscale Analysis")

    # Judgement Dimension Number Analysis (helpsteer2 and neurips)
    dimension_sizes = [1, 3, 5]
    for size in dimension_sizes:
        run_metadata = {'type' : 'base classifier', 'dimension' : size}
        model.getreport(data.get_dimensionized_features(train_data_base[0], 'helpsteer2', size), data.get_dimensionized_features(test_data_base[0], 'helpsteer2', size), 'helpsteer2', f"JudgementDimension_{size}")
        model.getreport(data.get_dimensionized_features(train_data_base[3], 'neurips', size), data.get_dimensionized_features(test_data_base[3], 'neurips', size), 'neurips', f"JudgementDimension_{size}")

    # With augmented features
    # Rating Scale Analysis (helpsteer2 and helpsteer3)
    run_metadata = {'type' : 'base classifier', 'rating_scale' : 'binary'}
    model.getreport(data.get_binarized_rating_scale(train_data[0], 'helpsteer2'), data.get_binarized_rating_scale(test_data[0], 'helpsteer2'), 'helpsteer2', "Ratingscale Analysis")
    model.getreport(data.get_binarized_rating_scale(train_data[1], 'helpsteer3'), data.get_binarized_rating_scale(test_data[1], 'helpsteer3'), 'helpsteer3', "Ratingscale Analysis")

    # Judgement Dimension Number Analysis (helpsteer2 and neurips)
    dimension_sizes = [1, 3, 5]
    for size in dimension_sizes:
        run_metadata = {'type' : 'base classifier', 'dimension' : size}
        model.getreport(data.get_dimensionized_features(train_data[0], 'helpsteer2', size), data.get_dimensionized_features(test_data[0], 'helpsteer2', size), 'helpsteer2', f"JudgementDimension_{size}")
        model.getreport(data.get_dimensionized_features(train_data[3], 'neurips', size), data.get_dimensionized_features(test_data[3], 'neurips', size), 'neurips', f"JudgementDimension_{size}")





                    




                    
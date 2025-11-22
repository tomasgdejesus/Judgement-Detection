import data as data
import mlmodel as model

import os
import time
import sys

# added group detection as option as it is time consuming
COMMANDS_OUT = \
"""Usage: python main.py <run_groups=y|n>
run_groups: run group detection if y, otherwise don't."""

if __name__ == "__main__":
    if os.path.exists('metrics.json'):
        # Just in case, to not overwrite data
        print("metrics.json already exists in current directory. Please move it and run the program again.")
        sys.exit(1)

    if len(sys.argv) != 2:
        print(COMMANDS_OUT)
        sys.exit(1)

    run_groups = sys.argv[1]
    if run_groups != 'y' and run_groups != 'n':
        print(COMMANDS_OUT)
        sys.exit(1)

    start_time = time.perf_counter()
    print("hi")

    # Build the dataset with only judgement decisions
    train_data, train_data_comp = data.getrawdata('train', 1)
    test_data, test_data_comp = data.getrawdata('test', 1)

    # For debug purpose
    # Generate csv files 
    data.write_to_csv(train_data)
    
    # Base classifier
    with open('metrics.txt', 'a') as f:
        f.write("\n Judgement Decision \n")
    run_metadata = {'type' : 'base classifier'}
    m_helpsteer2lr, m_helpsteer2rf, scaler_helpsteer2 = model.getreport(train_data[0], test_data[0], 'helpsteer2', run_metadata)
    m_helpsteer3lr, m_helpsteer3rf, scaler_helpsteer3 = model.getreport(train_data[1], test_data[1], 'helpsteer3', run_metadata)
    m_antiquelr, m_antiquerf, scaler_antique = model.getreport(train_data[2], test_data[2], 'antique', run_metadata)
    m_neuripslr, m_neuripsrf, scaler_neurips = model.getreport(train_data[3], test_data[3], 'neurips', run_metadata)
    modelslr = [m_helpsteer2lr, m_helpsteer3lr, m_antiquelr, m_neuripslr]
    modelsrf = [m_helpsteer2rf, m_helpsteer3rf, m_antiquerf, m_neuripsrf]
    scaler = [scaler_helpsteer2, scaler_helpsteer3, scaler_antique, scaler_neurips]

    # Rating Scale Analysis (helpsteer2 and helpsteer3)
    run_metadata = {'type' : 'base classifier', 'rating_scale' : 'binary'}
    model.getreport(data.get_binarized_rating_scale(train_data[0], 'helpsteer2'), data.get_binarized_rating_scale(test_data[0], 'helpsteer2'), 'helpsteer2', run_metadata)
    model.getreport(data.get_binarized_rating_scale(train_data[1], 'helpsteer3'), data.get_binarized_rating_scale(test_data[1], 'helpsteer3'), 'helpsteer3', run_metadata)

    # Judgement Dimension Number Analysis (helpsteer2 and neurips)
    dimension_sizes = [1, 3, 5]
    for size in dimension_sizes:
        run_metadata = {'type' : 'base classifier', 'dimension' : size}
        model.getreport(data.get_dimensionized_features(train_data[0], 'helpsteer2', size), data.get_dimensionized_features(test_data[0], 'helpsteer2', size), 'helpsteer2', run_metadata)
        model.getreport(data.get_dimensionized_features(train_data[3], 'neurips', size), data.get_dimensionized_features(test_data[3], 'neurips', size), 'neurips', run_metadata)

    print("Combining features...")
    # Build the dataset with augmented features (judgement + llm + linguistic)
    train_data = data.getcombine_features('train')
    test_data = data.getcombine_features('test')

    # For debug purpose
    # Generate csv files 
    data.write_to_csv(train_data)

    with open('metrics.txt', 'a') as f:
        f.write("\n Augmented Features \n")
    # Base classifier
    run_metadata = {'type' : 'augmented features'}
    model.getreport(train_data[0], test_data[0], 'helpsteer2', run_metadata)
    model.getreport(train_data[1], test_data[1], 'helpsteer3', run_metadata)
    model.getreport(train_data[2], test_data[2], 'antique', run_metadata)
    model.getreport(train_data[3], test_data[3], 'neurips', run_metadata)
    
    # Rating Scale Analysis (helpsteer2 and helpsteer3)
    run_metadata = {'type' : 'augmented features', 'rating_scale' : 'binary'}
    model.getreport(data.get_binarized_rating_scale(train_data[0], 'helpsteer2'), data.get_binarized_rating_scale(test_data[0], 'helpsteer2'), 'helpsteer2', run_metadata)
    model.getreport(data.get_binarized_rating_scale(train_data[1], 'helpsteer3'), data.get_binarized_rating_scale(test_data[1], 'helpsteer3'), 'helpsteer3', run_metadata)

    # Judgement Dimension Number Analysis (helpsteer2 and neurips)
    dimension_sizes = [1, 3, 5]
    for size in dimension_sizes:
        run_metadata = {'type' : 'augmented features', 'dimension' : size}
        model.getreport(data.get_dimensionized_features(train_data[0], 'helpsteer2', size), data.get_dimensionized_features(test_data[0], 'helpsteer2', size), 'helpsteer2', run_metadata)
        model.getreport(data.get_dimensionized_features(train_data[3], 'neurips', size), data.get_dimensionized_features(test_data[3], 'neurips', size), 'neurips', run_metadata)

    # Group Detection
    if run_groups == 'y':
        k = [2, 4, 8 ,16]
        for i in range(len(k)):
            # train_data, train_data_comp = data.getrawdata('train', k[i])
            test_data, test_data_comp = data.getrawdata('test', k[i])
            with open('metrics.txt', 'a') as f:
                f.write(f"\n Group size of {k[i]} \n")
            run_metadata = {'type' : f'group size of {k[i]}'}
            model.predict(test_data_comp, modelslr, modelsrf, scaler, run_metadata)
    else:
        print("Skipping group detection...")
        print(f"Time Elapsed: {time.perf_counter() - start_time}")

    print(f"Time Elapsed: {time.perf_counter() - start_time}")
                
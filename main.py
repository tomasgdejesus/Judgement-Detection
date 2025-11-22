import data as data
import mlmodel as model


if __name__ == "__main__":
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
    m_helpsteer2lr, m_helpsteer2rf, scaler_helpsteer2 = model.getreport(train_data[0], test_data[0], 'helpsteer2')
    m_helpsteer3lr, m_helpsteer3rf, scaler_helpsteer3 = model.getreport(train_data[1], test_data[1], 'helpsteer3')
    m_antiquelr, m_antiquerf, scaler_antique = model.getreport(train_data[2], test_data[2], 'antique')
    m_neuripslr, m_neuripsrf, scaler_neurips = model.getreport(train_data[3], test_data[3], 'neurips')
    modelslr = [m_helpsteer2lr, m_helpsteer3lr, m_antiquelr, m_neuripslr]
    modelsrf = [m_helpsteer2rf, m_helpsteer3rf, m_antiquerf, m_neuripsrf]
    scaler = [scaler_helpsteer2, scaler_helpsteer3, scaler_antique, scaler_neurips]

    # Build the dataset with augmented features (judgement + llm + linguistic)
    train_data = data.getcombine_features('train')
    test_data = data.getcombine_features('test')

    # For debug purpose
    # Generate csv files 
    data.write_to_csv(train_data)

    with open('metrics.txt', 'a') as f:
        f.write("\n Augmented Features \n")
    # Base classifier
    model.getreport(train_data[0], test_data[0], 'helpsteer2')
    model.getreport(train_data[1], test_data[1], 'helpsteer3')
    model.getreport(train_data[2], test_data[2], 'antique')
    model.getreport(train_data[3], test_data[3], 'neurips')
    

    # Group Detection
    k = [2, 4, 8 ,16]
    for i in range(len(k)):
        # train_data, train_data_comp = data.getrawdata('train', k[i])
        test_data, test_data_comp = data.getrawdata('test', k[i])
        with open('metrics.txt', 'a') as f:
            f.write(f"\n Group size of {k[i]} \n")
        model.predict(test_data_comp, modelslr, modelsrf, scaler)
                
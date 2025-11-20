import data as data
import mlmodel as model


if __name__ == "__main__":
    print("hi")

    # # Build the dataset
    # train_data, train_data_comp = data.builddata('train')
    # test_data, test_data_comp = data.builddata('test')

    # # Base classifier
    # train_data[0].to_csv('helpsteer2.csv', index=False)
    # train_data[1].to_csv('helpsteer3.csv', index=False)
    # train_data[2].to_csv('antique.csv', index=False)
    # train_data[3].to_csv('neurips.csv', index=False)
    # model.getreport(train_data[0], test_data[0], 'helpsteer2')
    # model.getreport(train_data[1], test_data[1], 'helpsteer3')
    # model.getreport(train_data[2], test_data[2], 'antique')
    # model.getreport(train_data[3], test_data[3], 'neurips')

    # augmented features
    train_data = data.getcombine_features('train')
    test_data = data.getcombine_features('test')

    # Base classifier
    train_data[0].to_csv('helpsteer2_combine.csv', index=False)
    train_data[1].to_csv('helpsteer3_combine.csv', index=False)
    train_data[2].to_csv('antique_combine.csv', index=False)
    train_data[3].to_csv('neurips_combine.csv', index=False)
    model.getreport(train_data[0], test_data[0], 'helpsteer2')
    model.getreport(train_data[1], test_data[1], 'helpsteer3')
    model.getreport(train_data[2], test_data[2], 'antique')
    model.getreport(train_data[3], test_data[3], 'neurips')
    

    
                
# Data Processing

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    #load the train.csv file into a dataframe
    train = pd.read_csv('./petfinder-adoption-prediction/train/train.csv')
    #load the test.csv file into a dataframe
    test = pd.read_csv('./petfinder-adoption-prediction/test/test.csv')

    #drop text/object features from train
    train = train.drop(['Name', 'RescuerID', 'Description','PetID'], axis = 1)
    #drop text/object features from test
    test = test.drop(['Name', 'RescuerID', 'Description','PetID'], axis = 1)

    #categorise the features into nominal (categorical) or numerical features
    nominal_features = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'State']
    numerical_features = ['Age', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
            'Sterilized', 'Health', 'Quantity', 'Fee', 'VideoAmt','PhotoAmt']

    #TODO currently the onehotencoder ignores the subfeatures in the test dataset that dont appear in the training dataset
    #need to find a way to include these subfeatures initially in the training dataset, or a better way to handle them

    #instantiate a OneHotEncoder that will output an array
    ohe = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
    #fit the encoder to the trainins dataset
    #will keep the exploded categories for the testing dataset
    ohe.fit(train[nominal_features])

    #encode the nominal features of the training dataset and replace the non-encoded data
    train_nominal_encoded = ohe.transform(train[nominal_features])
    train_nominal_encoded = pd.DataFrame(train_nominal_encoded)
    train_nominal_encoded.columns = ohe.get_feature_names_out(nominal_features)

    train.drop(nominal_features, axis=1, inplace = True)
    train = pd.concat([train, train_nominal_encoded], axis = 1)

    #encode the nominal features of the testing dataset and replace the non-encoded data
    test_nominal_encoded = ohe.transform(test[nominal_features])
    test_nominal_encoded = pd.DataFrame(test_nominal_encoded)
    test_nominal_encoded.columns = ohe.get_feature_names_out(nominal_features)

    test.drop(nominal_features, axis=1, inplace = True)
    test = pd.concat([test, test_nominal_encoded], axis = 1)

    #setup the scaler
    scaler = StandardScaler()

    #scale the numerical features of the train dataframe and replace the unscaled data
    train_numerical = train[numerical_features]
    train_numerical = scaler.fit_transform(train_numerical)
    train_numerical_df = pd.DataFrame(train_numerical, columns=numerical_features)

    train = train.drop(numerical_features, axis = 1)
    train = pd.concat([train, train_numerical_df], axis=1)

    #scale the numerical features of the test dataframe and replace the unscaled data
    test_numerical = test[numerical_features]
    test_numerical = scaler.fit_transform(test_numerical)
    test_numerical_df = pd.DataFrame(test_numerical, columns=numerical_features)

    test = test.drop(numerical_features, axis = 1)
    test = pd.concat([test, test_numerical_df], axis=1)

    X_train = train.drop(['AdoptionSpeed'],axis=1)
    y_train = train['AdoptionSpeed']
    X_predict = test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    
    return X_train, y_train, X_test, y_test, X_predict
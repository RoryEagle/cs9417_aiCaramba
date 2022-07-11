import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(train_file):
    #load the train.csv file into a dataframe
    df = pd.read_csv('./petfinder-adoption-prediction/train/train.csv')
    #drop text/object features from train
    df = df.drop(['Name', 'RescuerID', 'Description','PetID'], axis = 1)

    #categorise the features into nominal (categorical) or numerical features
    nominal_features = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'State']
    numerical_features = ['Age', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
        'Sterilized', 'Health', 'Quantity', 'Fee', 'VideoAmt','PhotoAmt']

    #split the training.csv dataframe into 70% training, 30% testing with a fixed random state
    train, test = train_test_split(df, test_size = 0.3, random_state=123)

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
    #reset the indices to make a merge possible without NaN values appearing
    train_nominal_encoded.reset_index(drop=True, inplace = True)
    train.reset_index(drop = True, inplace = True)
    #append the encoded nominal columns to the training set
    train = pd.concat([train, train_nominal_encoded], axis = 1)

    #encode the nominal features of the testing dataset and replace the non-encoded data
    test_nominal_encoded = ohe.transform(test[nominal_features])
    test_nominal_encoded = pd.DataFrame(test_nominal_encoded)
    test_nominal_encoded.columns = ohe.get_feature_names_out(nominal_features)

    test.drop(nominal_features, axis=1, inplace = True)
    #reset the indices to make a merge possible without NaN values appearing
    test_nominal_encoded.reset_index(drop=True, inplace = True)
    test.reset_index(drop = True, inplace = True)
    #append the encoded nominal columns to the testing set
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

    X_train = train.drop(['AdoptionSpeed'],axis=1).values
    Y_train = train['AdoptionSpeed'].values
    X_test = test.drop(['AdoptionSpeed'], axis = 1).values
    Y_test = test['AdoptionSpeed'].values
    
    return X_train, Y_train, X_test, Y_test
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

import xgboost as xgb

selected_features = ['GenHlth', 'BMI', 'HighBP', 'Age', 'HighChol', 'Income', 'DiffWalk', 
                         'PhysHlth', 'Education', 'HeartDiseaseorAttack']

xgb_params = {
    'eta': 0.08, 
    'max_depth': 3,
    'min_child_weight': 10,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

boost_rounds = 150

output_file = 'xgb_trained_model.bin'


#get x and y after removing duplicates
def transform_io(df):
    df_nodup = df[selected_features].copy()
    dup_idx = df[df_nodup.duplicated()].index
    x = df_nodup.drop_duplicates(keep='first')
    train_dicts = x.to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    df_y_nodup = df['Diabetes_binary'].copy()
    df_y_nodup.drop(dup_idx, inplace=True)
    y = df_y_nodup.values
    
    return X_train, y, dv

if __name__ == '__main__':
    
    df = pd.read_csv("cleaned_data.csv")
    print('Initialized data frame')
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    X_full_train, y_full_train, dv = transform_io(df_full_train)
    X_val, y_val, dv = transform_io(df_test)
    features = dv.get_feature_names_out()
    print('Set up testing and validation framework')
    dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names= list(features))
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(features))

    print('Starting model training')
    model = xgb.train(xgb_params, dtrain, num_boost_round=boost_rounds)
    print('Finished model training')
    y_pred = model.predict(dval)
    roc = roc_auc_score(y_val, y_pred)
    print('Finished validation')

    print('Roc value in test: %.3f' % (roc))


    # with is used to ensure that file is closed
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print('Saved model')

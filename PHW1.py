import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def FindBestModel(df, target, scalers = None, encoders = None, models = None) :
    """
    Find the best combination of scaler, encoder, fitting algoritm
    print best score and best combination
    Parameters
    --------------------------------
    df : preproessed Dataset
    target : target column 

    scalers: list of sclaer
            None: [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
            if you want to scale other ways, then put the sclaer in list
    encoders: list of encoder
        None: [OrdinalEncoder(), OneHotEncoder()]
        if you want to use only one, put a encoder in list
    models: list of encoder
        None: [DecisionTreeClassifier(entropy), DecisionTreeClassifier(gini), SVC(), LogisticRegression()]
        if you want to fitting other ways, then put in list
    """
    ##################################################
    """
    X: DataFrame to scaled
    y: DataFrame to encoding
    X_category : columns to encode
    X_num : columns to scaled

    df_cateEmpty is True if There is no category feature
    df_num is True if There is no numerical feature

    """
    X = df.drop(target, axis=1)
    y = df[target]

    X_category = X.select_dtypes(include = 'object')
    df_cateEmpty = X_category.empty
    X_num = X.select_dtypes(exclude = 'object')
    df_numEmpty = X_num.empty
    
    if scalers == None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: scale = scalers

    if encoders == None:
         encode = [OrdinalEncoder(), OneHotEncoder()]
    else: encode = encoders

    if models == None:
        DecisionTreeEntropy = DecisionTreeClassifier()
        DecisionTreeGini = DecisionTreeClassifier()
        SupportVC = SVC()
        LoR = LogisticRegression()
        classifier = [DecisionTreeEntropy, DecisionTreeGini, SupportVC, LoR]
    else: classifier = models
    

    # to Compare performance (accuracy) of the following classification
    # models against the same dataset.

    """
    best_score model name : best score
    best_param model name : best parameter
    best scaler_encoder : best scaler and encoder
    """

    best_score_DT_entropy = 0
    best_DT_entropy_param = []
    best_scaler_encoder_entropy = []

    best_score_DT_gini = 0
    best_DT_gini_param = []
    best_scaler_encoder_gini = []

    best_score_SVC = 0
    best_param_SVC = []
    best_scaler_encoder_SVC = []

    best_score_LR = 0
    best_param_LR = []
    best_scaler_encoder_LR = []

    # hyperparameters for GridSearchCV
    DTEparameters = {'criterion':["entropy"],'max_depth':[1,2,3], 'min_samples_split':[2,3]}
    DTGparameters = {'criterion':["gini"],'max_depth':[1,2,3], 'min_samples_split':[2,3]}
    LRparameters = {'C': [0.1, 1, 10],'penalty': ['l2']}
    SVCparameters = { 'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

    for i in scale :
        for j in encode:
            if df_numEmpty is False :
                scaler = i
                scaler = pd.DataFrame(scaler.fit_transform(X_num))

            if j == OrdinalEncoder() and df_cateEmpty is False:
                enc = j
                enc = enc.fit_transform(X_category)
                new_df = pd.concat([scaler,enc], axis=1)
            elif j == OneHotEncoder() and df_cateEmpty is False:
                dum = pd.DataFrame(pd.get_dummies(X_category))
                new_df = pd.concat([scaler, dum], axis=1)
            else:
                new_df = scaler

            for model in classifier:
                
                X_train, X_test, y_train, y_test = train_test_split(new_df,y,test_size=0.3, shuffle=True, random_state=34)
                print(type(X_train))
                
                # GridSearchCV of each model is used
                if model == DecisionTreeEntropy:
                    grid_dt_E = GridSearchCV(model, param_grid=DTEparameters, cv=3)
                    grid_dt_E.fit(X_train, y_train)
                    score = grid_dt_E.score(X_test, y_test)

                    if score > best_score_DT_entropy:
                        best_score_DT_entropy = score
                        best_DT_entropy_param = grid_dt_E.best_params_
                        best_scaler_encoder_entropy = [i,j]

                elif model == DecisionTreeGini:
                    grid_dt_G = GridSearchCV(model, param_grid=DTGparameters, cv=3)
                    grid_dt_G.fit(X_train, y_train)
                    score = grid_dt_G.score(X_test, y_test)

                    if score > best_score_DT_gini:
                        best_score_DT_gini = score
                        best_DT_gini_param = grid_dt_G.best_params_
                        best_scaler_encoder_gini = [i,j]

                elif model == SupportVC:
                    grid_SVC = GridSearchCV(model, param_grid=SVCparameters, cv=3)
                    grid_SVC.fit(X_train, y_train)
                    score = grid_SVC.score(X_test, y_test)

                    if score > best_score_SVC:
                        best_score_SVC = score
                        best_param_SVC = grid_SVC.best_params_
                        best_scaler_encoder_SVC = [i,j]
        
                else:
                    grid_LR = GridSearchCV(model, param_grid=LRparameters, cv=3)
                    grid_LR.fit(X_train, y_train)
                    score = grid_LR.score(X_test, y_test)

                    if score > best_score_LR:
                        best_score_LR = score
                        best_param_LR = grid_LR.best_params_
                        best_scaler_encoder_LR = [i,j]

    # print best score and hyperparameters of each model.
    print('Best score for Decision Tree (Entropy) :', best_score_DT_entropy)
    print('Model parameters : ', best_DT_entropy_param)
    print('Scaling and Encoding Method :', best_scaler_encoder_entropy)
    print()

    print('Best score for Decision Tree (GINI) :', best_score_DT_gini)
    print('Model parameters : ', best_DT_gini_param)
    print('Scaling and Encoding Method :', best_scaler_encoder_gini)
    print()

    print('Best score for Support Vector Machine :', best_score_SVC)
    print('Model parameters : ', best_param_SVC)
    print('Scaling and Encoding Method :', best_scaler_encoder_SVC)
    print()

    print('Best score for Logistic Regression :', best_score_LR)
    print('Model parameters : ', best_param_LR)
    print('Scaling and Encoding Method :', best_scaler_encoder_LR)

    return

df = pd.read_csv('ML\\breast-cancer-wisconsin.data')

df.columns = ['id','thickness','size_uniformity','shape_uniformity','adhesion','epithelial_size',
              'bare_nucleoli','bland_chromatin','normal_nucleoli','mitoses','class']

# describe dataset
print('Original dataset Row length : ',len(df))

# preprocessing
df['bare_nucleoli'] = df['bare_nucleoli'].astype('category')
idx = df[df['bare_nucleoli'] == '?'].index
df_temp = df.drop(idx)
print('preprocessing dataset Row length : ', len(df_temp))

# drop id column
df_temp = df_temp.drop(['id'], axis = 1)

FindBestModel(df_temp, 'class')

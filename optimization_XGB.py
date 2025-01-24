import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef

base = pd.read_csv("basePreProcessedAllAbFinalcsv")
y_data = base["status"]
x_data = base.drop(["status"], axis=1)

random_state = 1
folds = 10
countFold = 1

kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

def objective(trial):

    # Definindo os hiperparâmetros a serem otimizados
    learning_rate = trial.suggest_float("learning_rate", 0.025, 0.4, step=0.025)
    n_estimators = trial.suggest_int("n_estimators", 75, 500, step=25)
    max_depth = trial.suggest_int("max_depth", 6, 26, step=2)
    subsample = trial.suggest_float("subsample", 0.5, 1, step=0.1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1, step=0.1)
    gamma = trial.suggest_float("gamma", 0, 2, step=0.1)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 8, step=1)
    
    # Criando o modelo com os hiperparâmetros sugeridos
    classifier = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma = gamma,
        min_child_weight=min_child_weight,
        random_state=random_state
    )

    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    results = {'ACC-Balanced':[], 'F1-Macro':[], 'MCC': []}

    #Treinamendo do modelo com k-fold corss validation e aplicação do pipeline
    for train_index, test_index in kf.split(base):
        
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

        #Mean Imputation
        #x_train = imputer.fit_transform(x_train) #Uncomment this for pipeline 2
        #x_test = imputer.transform(x_test) #Uncomment this for pipeline 2

        # Standardize the sets using the same scaler
        #x_train = scaler.fit_transform(x_train)
        #x_test = scaler.transform(x_test)

        # Apply PCA to training data to retain 95% variance
        #pca = PCA(n_components=0.95) #Comment this for pipeline 1_semPCA
        #x_train = pca.fit_transform(x_train) #Comment this for pipeline 1_semPCA

        # Apply the same PCA transformation to the test set
        #x_test = pca.transform(x_test) #Comment this for pipeline 1_semPCA

        # SMOTE augmentation on the PCA-reduced training set
        #smote = SMOTE(k_neighbors=3, random_state=random_state)
        #x_train, y_train = smote.fit_resample(x_train, y_train)

        #x_train = pd.DataFrame(x_train)
        #x_test = pd.DataFrame(x_test)
        
        classifier.fit(x_train, y_train)
        y_pred_proba = classifier.predict_proba(x_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int) #Threshold de 0.5 considerado

        balanced_acc = balanced_accuracy_score(y_test, y_pred) #Se quisermos maximizar a acurácia balanceada
        f1_macro = f1_score(y_test, y_pred, average='macro') #Se quisermos maximizar o F1 macro
        mcc = matthews_corrcoef(y_test, y_pred) #Se quisermos maximizar o MCC

        results['ACC-Balanced'].append(balanced_acc)
        results['F1-Macro'].append(f1_macro)
        results['MCC'].append(mcc)
    
    return np.mean(results["ACC-Balanced"])  #Escolha da métrica a ser maximizada

study = optuna.create_study(study_name="XGB_200_pipeTest_none", sampler=optuna.samplers.CmaEsSampler(seed=random_state), pruner=optuna.pruners.NopPruner, direction="maximize")
study.optimize(objective, n_trials = 200, show_progress_bar=True, n_jobs=10) #Avaliar inclusão do direction

# Melhor resultado
print("Melhor conjunto de hiperparâmetros:", study.best_params)
print("Melhor acurácia:", study.best_value)
study.trials_dataframe().to_csv(f"Optimization_results_XGB_Pipeline_none.csv", index=False)
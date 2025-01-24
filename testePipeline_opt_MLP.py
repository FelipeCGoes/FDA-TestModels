import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Auxiliar functions
def accuracy_per_class(y_test, y_pred):
    """
    Calculate accuracy per class.

    Parameters
    ----------
    y_test : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.

    Returns
    -------
    acc_pos: float
        Accuracy for the positive class.
    acc_neg: float
        Accuracy for the negative class.
    acc: float
        Average accuracy.

    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    if (tp + fp) == 0:
        acc_pos = 0
    else:
        acc_pos = tp / (tp + fp)

    if (tn + fn) == 0:
        acc_neg = 0
    else:
        acc_neg = tn / (tn + fn)

    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc_pos, acc_neg, acc

def auc_eval(y_test, y_pred, positive = 1):
    """
    Calculate AUC per class.

    Parameters
    ----------
    y_test : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.
    positive : int
        Index of positive class.

    Returns
    -------
    auc : float
        Area under the ROC curve.

    """
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=positive)
    return metrics.auc(fpr, tpr)

def roc_calc_viz_pred(y_true, y_pred):
    viz = RocCurveDisplay.from_predictions(
                            y_true,
                            y_pred
                        )

    return viz.fpr, viz.tpr, viz.roc_auc

def positive_negative_rate(y_test, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensibivity = tp / (tp+fn)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    return round(sensibivity, 5), round(specificity, 5), round(accuracy,5) 



base = pd.read_csv("basePreProcessedAllAbFinal.csv")
y_data = base["status"]
x_data = base.drop(["status"], axis=1)

attrCount = x_data.shape[1]

random_state = 1
folds = 10
countFold = 1
rangeThreshold = [0.1, 0.81, 0.1]
cvResults = []

thresholdList = []
for thresh in np.arange(start=rangeThreshold[0], stop=rangeThreshold[1], step=rangeThreshold[2]):
    thresholdList.append(thresh)

print("cuda") if torch.cuda.is_available() else print("cpu")
global device 
device = "cuda" if torch.cuda.is_available() else "cpu"


model = nn.Sequential(nn.Linear(attrCount, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(32, 1),
                nn.Sigmoid()
                )

# Modelo
model = model.to(device)

# Otimizador
optimizer = optim.AdamW(model.parameters(), lr=0.00012295920205037024, weight_decay=6.268225479552331e-05)

#Loss
loss_fn = nn.BCELoss().to(device)

kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
#imputer = KNNImputer(n_neighbors=5)
imputer = SimpleImputer(strategy='mean')
#scaler = StandardScaler()
countFolds = 1

#Treinamendo do modelo com k-fold corss validation e aplicação do pipeline
for train_index, test_index in kf.split(base):
    x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

    print(f"Fold: {countFold}/{folds}")
    countFold += 1

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

    x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    # Treinamento
    num_epochs = 500
    model.to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=False) #pin_memory=True - Para GPU
    val_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False) #pin_memory=True - Para GPU

    best_val_loss = float('inf')
    patience = 100
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0
        y_preds, y_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze()
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item()
                y_preds.append(y_pred.cpu())
                y_true.append(y_batch.cpu())

        val_loss /= len(val_loader)
        y_preds = torch.cat(y_preds).round()
        y_true = torch.cat(y_true)
        val_accuracy = balanced_accuracy_score(y_true, y_preds)

        if val_loss < best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        print(epoch)

    # Avaliação final
    model.eval()
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            y_preds.append(y_pred.cpu())
            y_true.append(y_batch.cpu())

    y_pred_proba = torch.cat(y_preds)
    y_test = torch.cat(y_true)

    results = {'model_name': [], 'F1':[], 'F1-Macro':[], 'ROC':[], 'acc-class-1':[], 'acc-class-2':[], 'ACC':[], 'ACC-Balanced':[], 'TPR': [], 'FPR':[], 'AUC': [], 'THRE': [], 'SEN': [], 'SPE': [], 'MCC': []}

    for thresh in np.arange(start=rangeThreshold[0], stop=rangeThreshold[1], step=rangeThreshold[2]):

        y_pred = (y_pred_proba >= thresh).int()

        print(f'Threshold with {thresh}')
                
        acc_pos, acc_neg, acc = accuracy_per_class(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        f1_macro = f1_score(y_test, y_pred, average='macro') #Inclusão do F1 macro
        auc = auc_eval(y_test, y_pred)
        acc_class = [acc_pos, acc_neg]
        viz_fpr = 0
        viz_tpr = 0
        viz_auc = 0 # No pipeline original: viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_proba) -> Removido pois não são métricas utilizadas e por estarem imprimindo os gráficos
        sen,spe,_ = positive_negative_rate(y_test,y_pred)
        mcc = matthews_corrcoef(y_test, y_pred) #Inclusão do MCC
        balanced_acc = balanced_accuracy_score(y_test, y_pred) #Inclusão do balanced_accuracy

        results['model_name'].append('Decision Tree')
        results['acc-class-1'].append(acc_pos)
        results['acc-class-2'].append(acc_neg)
        results['ACC'].append(acc)
        results['ACC-Balanced'].append(balanced_acc)
        results['F1'].append(f1)
        results['F1-Macro'].append(f1_macro)
        results['ROC'].append(auc)
        results['FPR'].append(viz_fpr)
        results['TPR'].append(viz_tpr)
        results['AUC'].append(viz_auc)
        results['THRE'].append(thresh)
        results['SPE'].append(spe)
        results['SEN'].append(sen)
        results['MCC'].append(mcc)

    cvResults.append(results) #Cada lista terá multiplos valores pois executa sobre varios thresholds testados (teria um valor só se fosse um único threshold). Assim, cvResults é um vetor de dicts

cvFinalMetrics = {"ACC-Balanced":[], "ACC":[], 'F1':[], 'F1-Macro':[], 'ROC':[], 'MCC': [], 'acc-class-1':[], 'acc-class-2':[], 'SEN': [], 'SPE': []} #Removido 'model_name', 'THRE', 'TPR', 'FPR' e 'AUC'. 'TPR', 'FPR' e 'AUC' foram removidos pois são vetores com mais de 500 valores, correspondendo a pontos da curva ROC discretizada, permite compor visualmente a curva ROC. 'THRE' foi removido por ser o Threshold. Foi reordenado para facilitar a adição na planilia
    
#Calculate average for each metric and each threshold
for metric in cvFinalMetrics:
    temp_metric = []

    for i in range(folds): #Pega o vetor de valores por fold da métrica em questão (são vetores pois temos um valor por threshold) 
        temp_metric.append(cvResults[i][metric])
    
    for i in range(len(thresholdList)): #Calcula a média por threshold
        aux = []
        
        for i2 in range(folds):
            aux.append(temp_metric[i2][i])
        
        cvFinalMetrics[metric].append(np.average(aux))
    
results_df = {"Threshold": thresholdList}
results_df.update(cvFinalMetrics)
results_df = pd.DataFrame(results_df)
print(results_df)
results_df.to_csv(f"FDA_FinalMLP_PipeNone_results.csv", index=False)

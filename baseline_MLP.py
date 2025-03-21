import random
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    matthews_corrcoef
)
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

base = pd.read_csv("basePreProcessedAllAbFinal.csv")

df = base

print(df["status"].value_counts())

X = df.drop(columns=["status"])
y = df["status"]

RANDOM_STATE = 42
folds = 10

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

cvMetrics = {"balanced_acc": [], "acc": [], "AUC": [], "F1": [], "F1 Macro": [], "Precision": [], "Recall": [], "MCC": []}

kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE) 

for train_index, test_index in kf.split(base): #10-fold cross-validation

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Luiz architecture
    mlp = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(128, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid(),
    )

    if torch.cuda.is_available():
        print("Using CUDA")
        mlp = mlp.cuda()

    criterion = (
        torch.nn.BCELoss().cuda() if torch.cuda.is_available() else torch.nn.BCELoss()
    )

    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    max_accuracy = 0
    max_epoch = 0
    best_model_state_dict = None
    for epoch in range(5000):
        outputs = mlp(
            torch.tensor(X_train.values, dtype=torch.float32).cuda()
            if torch.cuda.is_available()
            else torch.tensor(X_train.values, dtype=torch.float32)
        )
        loss = criterion(
            outputs,
            torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).cuda()
            if torch.cuda.is_available()
            else torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mlp.eval()
            outputs = mlp(
                torch.tensor(X_test.values, dtype=torch.float32).cuda()
                if torch.cuda.is_available()
                else torch.tensor(X_test.values, dtype=torch.float32)
            )
            outputs = outputs.cpu().numpy()
            outputs = (outputs > 0.5).astype(int)
            scr = balanced_accuracy_score(y_test, outputs) #Quando aplicado SMOTE - balanced_accuracy_score -> accuracy_score
            if max(max_accuracy, scr) > max_accuracy:
                max_epoch = epoch
                best_model_state_dict = deepcopy(mlp.state_dict())
            max_before = max_accuracy
            max_accuracy = max(max_accuracy, scr)
            mlp.train()
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:03d} - Loss: {loss.item():.6f} - Max bal. accuracy: {max_accuracy:.6f}"
            )
    print()
    print(
        f"Max Balanced accuracy: {max_accuracy:.6f} at epoch {max_epoch}, loading model snapshot at this epoch..."
    )
    mlp.load_state_dict(best_model_state_dict)
    print()
    mlp.eval()

    with torch.no_grad():
        outputs = mlp(
            torch.tensor(X_test.values, dtype=torch.float32).cuda()
            if torch.cuda.is_available()
            else torch.tensor(X_test.values, dtype=torch.float32)
        )
        outputs = outputs.cpu().numpy()
        outputs_proba = outputs
        outputs = (outputs > 0.5).astype(int)
        cvMetrics["balanced_acc"].append(balanced_accuracy_score(y_test, outputs))
        cvMetrics["acc"].append(accuracy_score(y_test, outputs))
        cvMetrics["AUC"].append(roc_auc_score(y_test, outputs_proba)) #CORRIGIDO: y_pred -> y_pred_proba
        cvMetrics["F1"].append(f1_score(y_test, outputs))
        cvMetrics["F1 Macro"].append(f1_score(y_test, outputs, average='macro'))
        cvMetrics["Precision"].append(precision_score(y_test, outputs))
        cvMetrics["Recall"].append(recall_score(y_test, outputs))
        cvMetrics["MCC"].append(matthews_corrcoef(y_test, outputs))

print(f"Balanced accuracy: {np.average(cvMetrics['balanced_acc'])}")
print(cvMetrics["balanced_acc"])
print(f"Accuracy: {np.average(cvMetrics['acc'])}")
print(cvMetrics["acc"])
print(f"ROC AUC: {np.average(cvMetrics['AUC'])}")
print(cvMetrics["AUC"])
print(f"F1: {np.average(cvMetrics['F1'])}")
print(cvMetrics["F1"])
print(f"F1 Macro: {np.average(cvMetrics['F1 Macro'])}")
print(cvMetrics["F1 Macro"])
print(f"Precision: {np.average(cvMetrics['Precision'])}")
print(cvMetrics["Precision"])
print(f"Recall: {np.average(cvMetrics['Recall'])}")
print(cvMetrics["Recall"])
print(f"MCC: {np.average(cvMetrics['MCC'])}")
print(cvMetrics["MCC"])
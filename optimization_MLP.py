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

base = pd.read_csv("basePreProcessedAllAbFinal.csv")
y_data = base["status"]
x_data = base.drop(["status"], axis=1)

attrCount = x_data.shape[1]

random_state = 1
folds = 10
countFold = 1

print("cuda") if torch.cuda.is_available() else print("cpu")
global device 
device = "cuda" if torch.cuda.is_available() else "cpu"

def createMLP(input_dim, layers_config, activation_fn, use_batch_norm, use_dropout, output_activation):
    layers = []
    for i in range(len(layers_config)):
        layers.append(nn.Linear(input_dim if i == 0 else layers_config[i - 1], layers_config[i]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(layers_config[i]))
        layers.append(activation_fn)
        if use_dropout:
            layers.append(nn.Dropout(p=0.5))
    layers.append(nn.Linear(layers_config[-1], 1))  # Output layer
    layers.append(output_activation)  # Binary classification
    
    return nn.Sequential(*layers)

imputer = SimpleImputer(strategy='mean')
kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

def objective(trial):

    num_layers = trial.suggest_int("num_layers", 1, 4)
    layers_config = [trial.suggest_categorical(f"n_units_layer_{i}", [32, 64, 128, 256, 512]) for i in range(num_layers)]
    activation_fn_name = trial.suggest_categorical("activation", ["ReLU", "Tanh", "GELU"])
    activation_fn = getattr(nn, activation_fn_name)()
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    use_dropout = trial.suggest_categorical("use_dropout", [True, False])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)  #Regularização L2
    loss_fn = nn.BCELoss().to(device)
    output_activation = nn.Sigmoid()

    # Modelo
    model = createMLP(attrCount, layers_config, activation_fn, use_batch_norm, use_dropout, output_activation)
    model = model.to(device)

    # Otimizador
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    #imputer = KNNImputer(n_neighbors=5)
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

        x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
        y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

        # Treinamento
        num_epochs = 500
        model.to(device)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=False, pin_memory=True)
        val_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False, pin_memory=True)

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

            if val_loss < best_val_loss:
                best_val_loss = min(best_val_loss, val_loss)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

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
        y_pred = (y_pred_proba >= 0.5).int() #Threshold de 0.5 considerado
        y_test = torch.cat(y_true)

        balanced_acc = balanced_accuracy_score(y_test, y_pred) #Se quisermos maximizar a acurácia balanceada
        f1_macro = f1_score(y_test, y_pred, average='macro') #Se quisermos maximizar o F1 macro
        mcc = matthews_corrcoef(y_test, y_pred) #Se quisermos maximizar o MCC

        results['ACC-Balanced'].append(balanced_acc)
        results['F1-Macro'].append(f1_macro)
        results['MCC'].append(mcc)
    
    return np.mean(results["ACC-Balanced"])  #Escolha da métrica a ser maximizada

study = optuna.create_study(study_name="MLP_200_pipeTest_None", sampler=optuna.samplers.TPESampler(seed=random_state), pruner=optuna.pruners.NopPruner, direction="maximize", storage="sqlite:///MLP_200_pipeTest_None.db", load_if_exists=True)
study.optimize(objective, n_trials = 200, show_progress_bar=True, n_jobs=12)

# Melhor resultado
print("Melhor conjunto de hiperparâmetros:", study.best_params)
print("Melhor acurácia balanceada:", study.best_value)
study.trials_dataframe().to_csv(f"Optimization_results_MLP_Pipeline_None.csv", index=False)
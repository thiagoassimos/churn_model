

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score


def save_figure(fig, figure_name):
    """Salva a figura na pasta output/figures."""
    # Caminho absoluto para a raiz do projeto (onde está o diretório 'output')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Caminho para a raiz do projeto
    output_dir = os.path.join(project_root, 'output', 'figures')  # Caminho absoluto para 'output/figures'

    fig_path = os.path.join(output_dir, figure_name)

    try:
        fig.savefig(fig_path)
        print(f"Figura salva em: {fig_path}")
    except Exception as e:
        print(f"Erro ao salvar a figura: {e}")


def train_xgb_model(X_train, y_train):
    """Treina um modelo XGBoost com busca aleatória de parâmetros."""
    xgb = Pipeline([('scaler', MinMaxScaler()), ('model', XGBClassifier(random_state=10))])
    
    params_xgb = {
        'model__n_estimators': [100, 300, 500],
        'model__learning_rate': [0.01, 0.1],
        'model__max_depth': [2, 13],
        'model__subsample': [0.7, 0.9, 1],
        'model__colsample_bytree': [0.7, 0.9, 1],
        'model__gamma': [0, 0.1, 0.3],
        'model__reg_alpha': [0, 0.01, 0.1],
        'model__reg_lambda': [0, 0.01, 0.1]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
    random_search_xgb = RandomizedSearchCV(
        estimator=xgb, param_distributions=params_xgb, n_iter=10, 
        scoring='f1_weighted', cv=cv, refit=True, error_score=0, n_jobs=-1
    )
    random_search_xgb.fit(X_train, y_train)
    return random_search_xgb.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Avalia a performance do modelo."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_pred_proba > 0.25).astype(int)

    precision = precision_score(y_test, y_pred_adjusted)
    recall = recall_score(y_test, y_pred_adjusted)
    f1 = f1_score(y_test, y_pred_adjusted)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC: {roc_auc:.3f}\n")

    # Plotando a curva ROC com base nas taxas de falso positivo e verdadeiro positivo
    fig1 = plt.figure(figsize=(8, 6)) 
    tpr_list= []
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
   
    fprs = np.linspace(0, 1, 100)
    tprs = np.interp(fprs, fpr, tpr)
    tprs[0] = 0
    tpr_list.append(tprs)

    mean_tpr_xgb = np.mean(tpr_list, axis=0)
    mean_auc_xgb = auc(fprs, mean_tpr_xgb)

    plt.plot(fprs, mean_tpr_xgb, color='red', label=f'XGBoosting (AUC = {mean_auc_xgb:.2f})', lw=2)

    plt.title('Curva ROC e Área sob a curva (AUC)')
    plt.plot([0, 1], [0, 1], 'k--', label='Classificador Aleatório')
    plt.plot([0, 0], [0, 1],  label='Modelo Perfeito',color='green')
    plt.plot([0, 1], [1, 1],  label='Modelo Perfeito',color='green')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.legend(loc='lower right')
    plt.legend(fontsize="8")
    save_figure(fig1, 'roc_curve_xgb.png')                  
    plt.show()


    # Plotando a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
    fig2 = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, cmap='YlGnBu', annot=True, fmt='g')
    ax.set_xlabel('Predição')
    ax.set_ylabel('Realidade')
    ax.set_title('Matriz de Confusão')
    save_figure(fig2, 'conf_matrix.png')
    plt.show()

    
    return precision, recall, f1, roc_auc








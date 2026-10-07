"""
Atividade Prática: Modelagem de Risco de Crédito
Avaliação de Inadimplência com Random Forest & Análise Crítica de Métricas
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def titulo(texto):
    print("\n" + "=" * 60)
    print(texto)
    print("=" * 60)


# ============================================================
# PARTE 1 - Implementação do Modelo Base
# ============================================================
titulo("PARTE 1 - Modelo Base")

# Configuração de semente para reproducibilidade
np.random.seed(42)
n = 2000

# Criação do conjunto de dados sintético
dados = pd.DataFrame(
    {
        "RendaMensal": np.random.exponential(5000, n) + 1500,
        "ScoreCredito": np.random.randint(300, 950, n),
        "EmprestimosAtivos": np.random.poisson(1.5, n),
    }
)

# Cálculo da probabilidade de inadimplência e rotulagem
prob = 1 / (
    1
    + np.exp(
        -(
            -0.0003 * dados["RendaMensal"]
            - 0.005 * dados["ScoreCredito"]
            + 0.8 * dados["EmprestimosAtivos"]
        )
    )
)
dados["Inadimplente"] = (prob > np.random.uniform(0, 1, n)).astype(int)

# Divisão dos dados em treino e teste
X = dados.drop(columns=["Inadimplente"])
y = dados["Inadimplente"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treinamento do modelo RandomForest
modelo_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    max_features="sqrt",
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)
modelo_rf.fit(X_tr, y_tr)

# Previsões e avaliação
preds = modelo_rf.predict_proba(X_te)[:, 1]

print(f"Acurácia OOB (Treino): {modelo_rf.oob_score_:.4f}")
print(f"ROC-AUC no Teste: {roc_auc_score(y_te, preds):.4f}")


# ============================================================
# PARTE 2 - Importância das Variáveis
# ============================================================
titulo("PARTE 2 - Importância das Variáveis")

importancias = pd.Series(
    modelo_rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

print(importancias.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\nVariável de maior impacto: {importancias.index[0]}")

# Confirmação com Permutation Importance (queda de ROC-AUC no teste ao embaralhar
# cada variável), que não tem o viés da importância Gini por variáveis contínuas
perm = permutation_importance(
    modelo_rf, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=42, n_jobs=-1
)
importancias_perm = pd.Series(perm.importances_mean, index=X.columns).sort_values(
    ascending=False
)
print("\nPermutation Importance (queda de ROC-AUC no teste):")
print(importancias_perm.to_string(float_format=lambda v: f"{v:.4f}"))


# ============================================================
# PARTE 3 - Desbalanceamento e Métricas
# ============================================================
titulo("PARTE 3 - Desbalanceamento de Classes")

proporcao = y.mean()
print(f"Inadimplentes (classe 1): {y.sum()} de {len(y)} = {proporcao:.2%}")
print(f"Adimplentes   (classe 0): {(y == 0).sum()} de {len(y)} = {1 - proporcao:.2%}")

print(
    f"Baseline 'sempre adimplente' no teste: acurácia de {(y_te == 0).mean():.2%}"
)

y_pred = modelo_rf.predict(X_te)  # threshold padrão = 0.5

print("\nMatriz de Confusão (teste):")
print(
    pd.DataFrame(
        confusion_matrix(y_te, y_pred),
        index=["Real 0", "Real 1"],
        columns=["Prev 0", "Prev 1"],
    )
)

print("\nRelatório de Classificação (teste):")
print(classification_report(y_te, y_pred, digits=4, zero_division=0))


# ============================================================
# PARTE 4 - Desafio Extra: class_weight e threshold
# ============================================================
titulo("PARTE 4 - class_weight='balanced' e ajuste de threshold")

modelo_bal = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
modelo_bal.fit(X_tr, y_tr)
preds_bal = modelo_bal.predict_proba(X_te)[:, 1]

# Compara Recall da classe 1 em diferentes cenários
cenarios = []
for nome, probas in [("Original", preds), ("Balanced", preds_bal)]:
    for limiar in [0.5, 0.3, 0.2, 0.1]:
        previsto = (probas >= limiar).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, previsto).ravel()
        cenarios.append(
            {
                "Modelo": nome,
                "Threshold": limiar,
                "Recall_1": recall_score(y_te, previsto, zero_division=0),
                "Detectados (TP)": tp,
                "Perdidos (FN)": fn,
                "Falsos alarmes (FP)": fp,
            }
        )

resultado = pd.DataFrame(cenarios)
print(resultado.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(f"\nROC-AUC modelo balanced: {roc_auc_score(y_te, preds_bal):.4f}")

resultado.to_csv("resultados_recall.csv", index=False)
print("\nTabela salva em resultados_recall.csv")

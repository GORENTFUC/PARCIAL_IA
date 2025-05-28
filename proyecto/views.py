# Se importan las librerias para el template y los renders
from django.shortcuts import render
import json
from pathlib import Path

# Librerias para operaciones matemáticas
import numpy as np
# Libreria para el manejo de datos
import pandas as pd

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Libreria para cambio de datos
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Libreria para balanceo de los datos
from sklearn.utils import resample
# Libreria para separar los datos de entrenamiento y pruebas
from sklearn.model_selection import train_test_split
# Librerias para los modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Libreria para las métricas
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# Libreria para optimización de hiperparámetros
from sklearn.model_selection import GridSearchCV

# Ignorar warnings
import warnings
warnings.simplefilter('ignore')

def main(request):
    # Vista inicial sin predicciones
    return render(request, 'index.html', context={
        'show_initial': True  # Flag para mostrar vista inicial
    })

def prediccion(request):
    if request.method != 'POST':
        return main(request)

    ### Se cargan los datos
    data = pd.read_csv(BASE_DIR / 'proyecto' / 'data' / 'Dataset of Diabetes.csv')

    ### Preprocesamiento de datos
    # Eliminar la columna ID y No_Pation ya que no son relevantes para la predicción
    data = data.drop(['ID', 'No_Pation'], axis=1)
    
    # Convertir género a valores numéricos
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    
    # Separar características (X) y variable objetivo (y)
    X = data.drop('CLASS', axis=1)
    y = data['CLASS']
    
    # Normalizar las características numéricas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Se separan los datos (80% entrenamiento y 20% pruebas)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.6, random_state=42)

    # Balancear las clases usando resample
    df_train = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name='CLASS')], axis=1)
    
    # Identificar la clase mayoritaria
    class_counts = df_train['CLASS'].value_counts()
    majority_count = class_counts.max()
    
    # Balancear cada clase
    balanced_dfs = []
    for class_label in class_counts.index:
        df_class = df_train[df_train['CLASS'] == class_label]
        if len(df_class) < majority_count:
            df_class_balanced = resample(df_class,
                                      replace=True,
                                      n_samples=majority_count,
                                      random_state=42)
            balanced_dfs.append(df_class_balanced)
        else:
            balanced_dfs.append(df_class)
    
    # Combinar todos los datos balanceados
    df_train_balanced = pd.concat(balanced_dfs)
    
    # Separar características y objetivo de los datos balanceados
    X_train_balanced = df_train_balanced.drop('CLASS', axis=1)
    y_train_balanced = df_train_balanced['CLASS']

    ### Entrenar y predecir con múltiples modelos
    # Árbol de Decisión con GridSearch
    param_grid_dt = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
    dt.fit(X_train_balanced, y_train_balanced)
    predict_dt = dt.predict(X_test)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_balanced, y_train_balanced)
    predict_nb = nb.predict(X_test)

    # Support Vector Machine con GridSearch
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5)
    svm.fit(X_train_balanced, y_train_balanced)
    predict_svm = svm.predict(X_test)

    # Calcular métricas para los modelos
    def calculate_metrics(y_true, y_pred):
        return {
            'accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
            'precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=1) * 100, 2),
            'recall': round(recall_score(y_true, y_pred, average='weighted', zero_division=1) * 100, 2),
            'f1': round(f1_score(y_true, y_pred, average='weighted', zero_division=1) * 100, 2)
        }

    dt_metrics = calculate_metrics(y_test, predict_dt)
    nb_metrics = calculate_metrics(y_test, predict_nb)
    svm_metrics = calculate_metrics(y_test, predict_svm)

    # Crear datos para las gráficas de comparación individual
    comparison_data = {
        'indices': list(range(len(y_test))),
        'real_values': [1 if val == 'Y' else 0 for val in y_test.tolist()],
        'dt_predictions': [1 if val == 'Y' else 0 for val in predict_dt.tolist()],
        'nb_predictions': [1 if val == 'Y' else 0 for val in predict_nb.tolist()],
        'svm_predictions': [1 if val == 'Y' else 0 for val in predict_svm.tolist()]
    }

    # Preparar el contexto
    context = {
        'show_initial': False,  # Flag para mostrar resultados
        # Métricas del árbol de decisión
        'dt_accuracy': dt_metrics['accuracy'],
        'dt_precision': dt_metrics['precision'],
        'dt_recall': dt_metrics['recall'],
        'dt_f1': dt_metrics['f1'],
        
        # Métricas de Naive Bayes
        'nb_accuracy': nb_metrics['accuracy'],
        'nb_precision': nb_metrics['precision'],
        'nb_recall': nb_metrics['recall'],
        'nb_f1': nb_metrics['f1'],

        # Métricas de SVM
        'svm_accuracy': svm_metrics['accuracy'],
        'svm_precision': svm_metrics['precision'],
        'svm_recall': svm_metrics['recall'],
        'svm_f1': svm_metrics['f1'],
        
        # Valores para gráficas de métricas
        'dt_metrics_values': json.dumps([dt_metrics['accuracy'], dt_metrics['precision'], dt_metrics['recall'], dt_metrics['f1']]),
        'nb_metrics_values': json.dumps([nb_metrics['accuracy'], nb_metrics['precision'], nb_metrics['recall'], nb_metrics['f1']]),
        'svm_metrics_values': json.dumps([svm_metrics['accuracy'], svm_metrics['precision'], svm_metrics['recall'], svm_metrics['f1']]),
        
        # Datos para gráficas de comparación individual
        'comparison_data': json.dumps(comparison_data),
        
        # Mejores parámetros encontrados
        'dt_best_params': dt.best_params_,
        'svm_best_params': svm.best_params_
    }

    return render(request, 'index.html', context)

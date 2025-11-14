# Repositorio de Ejercicios de Python - Victor Guzman

Este repositorio contiene una colección de notebooks de Jupyter con ejercicios, exámenes parciales y finales de cursos de Python, cubriendo desde conceptos básicos hasta temas avanzados de análisis de datos y machine learning.

## 📋 Contenido

### 1. Python Básico - Examen Final
**Archivo:** `Victor Guzman - XIV_python_basico_Final.ipynb`

#### Temas cubiertos:
- **Pregunta 1 (10 puntos):** Extracción de información de bases de datos NoSQL
  - Manejo de diccionarios anidados
  - Consultas dinámicas de datos
  - Validación de campos y manejo de valores faltantes

- **Pregunta 2 (5 puntos):** Sistema de evaluación de perfil
  - Implementación de validación de respuestas
  - Lógica de decisión basada en múltiples criterios
  - Manejo de entrada de usuario

- **Pregunta 3 (5 puntos):** Sistema de recomendación de productos
  - Estructuras condicionales (if/elif/else)
  - Matching de patrones
  - Manejo de strings

#### Características destacadas:
```python
# Ejemplo: Extracción de datos de base NoSQL
def extraer_informacion(base_de_datos):
    num_columnas = int(input("Ingrese la cantidad de columnas a extraer: "))
    campos_solicitados = []
    
    for i in range(num_columnas):
        campo = input(f"Ingrese el nombre del campo {i + 1}: ")
        campos_solicitados.append(campo)
    
    # Extracción y formato de datos
    resultados = []
    for documento in base_de_datos:
        resultado_documento = {}
        for campo in campos_solicitados:
            valor = documento.get(campo, '-')
            resultado_documento[campo] = valor
        
        if any(valor != '-' for valor in resultado_documento.values()):
            resultados.append(resultado_documento)
```

---

### 2. Python Básico - Examen Parcial
**Archivo:** `Victor Guzman - XIV_python_basico_parcial.ipynb`

#### Ejercicios:
- **Pregunta 1 (6 puntos):** Bucle acumulador
  - Acumulación de números hasta alcanzar una suma objetivo
  - Control de flujo con while

- **Pregunta 2 (7 puntos):** Sistema de evaluación de notas
  - Condicionales múltiples
  - Evaluación de rendimiento académico

- **Pregunta 3 (7 puntos):** Enmascaramiento de datos personales
  - Manipulación de strings
  - Protección de información sensible
  - Formateo de salida

#### Ejemplo destacado:
```python
# Enmascaramiento de datos personales
nombre = input("Ingrese su nombre: ")
correo = input("Ingrese su correo: ")
telefono = input("Ingrese su número de teléfono: ")

# Mostrar parcialmente los datos
nombre_par = nombre[0] + "*" * (len(nombre) - 1)
telefono_par = telefono[:3] + "*" * (len(telefono) - 3)
correo_par = correo[:3] + "*" * (correo.index('@') - 3) + correo[correo.index('@'):]

mensaje = f"Estimado {nombre_par}, se registró su número {telefono_par} y su correo {correo_par}"
print(mensaje)
```

---

### 3. Análisis de Datos con Pandas - Examen Parcial
**Archivo:** `Victor Guzman - XIV_Python_parcial.ipynb`

#### Dataset: Encuesta Nacional de Hogares - Módulo de Empleo

**Librerías utilizadas:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

#### Ejercicios realizados:

**Pregunta 1 (5 puntos):** Cálculo de tasa de desempleo
- Agrupación de datos por año y género
- Cálculo de tasas porcentuales
- Análisis de tendencias temporales

**Pregunta 2 (7 puntos):** Función de rango personalizada
- Definición de funciones de agregación
- Uso de pivot tables
- Análisis estadístico (max - min)

**Pregunta 3 (5 puntos):** Cálculo de moda por grupos
- Estadísticas descriptivas
- Agrupación multinivel
- Análisis de edad por ocupación

**Pregunta 4 (3 puntos):** Exportación a Excel
- Múltiples hojas en un mismo archivo
- Formato de datos
- Uso de `ExcelWriter`

#### Código destacado:
```python
# Cálculo de tasa de desempleo por año y género
data['desempleo'] = (data['ocupado'] == 0).astype(int)

grouped_desempleo = data.groupby(['año', 'p207']).agg({
    'desempleo': ['sum', 'count']
})

grouped_desempleo['tasa_desempleo'] = (
    grouped_desempleo['sum'] / grouped_desempleo['count']
) * 100
```

---

### 4. Python Avanzado - Machine Learning
**Archivo:** `Victor Guzman - 2402_PYTHON_AVANZADO_PARCIAL.ipynb`

#### Dataset: MROZ (Wooldridge)
Análisis del mercado laboral de mujeres casadas con 753 observaciones y 22 variables.

**Librerías de ML:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
```

#### Pregunta 1: Modelo de Regresión Lineal
**Objetivo:** Predecir salarios (wage)

**Características:**
- División de datos (80/20)
- Normalización con StandardScaler
- Manejo de valores NaN con SimpleImputer
- Métricas: MSE y R²

**Resultados:**
```
Mean Squared Error (MSE): 2.31
R-squared (R²): 0.588
```

**Visualizaciones:**
- Scatter plot: Predicciones vs Valores Reales
- Histograma de residuos

#### Pregunta 2: Modelo de Clasificación Logística
**Objetivo:** Predecir participación laboral (inlf)

**Métricas evaluadas:**
- Accuracy: 0.715
- Confusion Matrix
- Classification Report
- ROC Curve (AUC)
- Precision-Recall Curve

**Visualizaciones:**
- Matriz de confusión (heatmap)
- Curva ROC
- Curva Precision-Recall

#### Código de ejemplo:
```python
# Modelo de Regresión Lineal
X = mroz_data.drop('wage', axis=1)
y = mroz_data['wage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## 🛠️ Requisitos y Dependencias

### Librerías principales:
```bash
pandas>=1.5.3
numpy>=1.23.5
matplotlib>=3.7.1
seaborn>=0.12.2
scikit-learn>=1.2.2
wooldridge>=0.4.4
```

### Instalación:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn wooldridge
```

---

## 📊 Estructura de Datos

### Base de datos NoSQL (Python Básico)
```python
base = {
    'usuario1': {
        'nombre': 'sergio',
        'apellido': 'diaz',
        'pwd': 'dth4',
        'telefono': '1234',
        'carrera': 'economia',
        'nivel_python': 'intermedio'
    },
    # ... más usuarios
}
```

### Dataset MROZ (Python Avanzado)
**Variables principales:**
- `inlf`: Participación en fuerza laboral (binaria)
- `wage`: Salario por hora
- `educ`: Años de educación
- `exper`: Años de experiencia laboral
- `age`: Edad
- `kidslt6`: Número de hijos menores de 6 años
- `kidsge6`: Número de hijos de 6-18 años

---

## 🎯 Conceptos Clave Aprendidos

### Python Básico
- ✅ Estructuras de control (if/elif/else, while, for)
- ✅ Manipulación de diccionarios y listas
- ✅ Entrada y salida de datos
- ✅ Funciones personalizadas
- ✅ Validación de datos

### Análisis de Datos
- ✅ Importación y limpieza de datos
- ✅ Merge y join de DataFrames
- ✅ Pivot tables y agregaciones
- ✅ Funciones de agregación personalizadas
- ✅ Exportación a Excel con múltiples hojas

### Machine Learning
- ✅ Regresión lineal
- ✅ Regresión logística
- ✅ Preprocesamiento de datos (normalización, imputación)
- ✅ División train/test
- ✅ Evaluación de modelos (MSE, R², accuracy, confusion matrix)
- ✅ Visualización de resultados

---

## 📈 Resultados y Métricas

### Modelo de Regresión Lineal
| Métrica | Valor |
|---------|-------|
| MSE | 2.31 |
| R² | 0.588 |

### Modelo de Clasificación
| Métrica | Valor |
|---------|-------|
| Accuracy | 71.5% |
| Precision (clase 1) | 0.73 |
| Recall (clase 1) | 0.79 |
| F1-Score (clase 1) | 0.76 |

---

## 🚀 Uso

### Ejecutar notebooks:
```bash
jupyter notebook "Victor Guzman - XIV_python_basico_Final.ipynb"
```

### Ejecutar en Google Colab:
Los notebooks están optimizados para Google Colab y pueden ejecutarse directamente subiendo los archivos.

---

## 📝 Notas Importantes

1. **Manejo de datos faltantes:** Se utiliza el símbolo '-' para representar valores ausentes en bases NoSQL
2. **Normalización:** Es crucial normalizar los datos antes de entrenar modelos de ML
3. **Validación cruzada:** Los modelos utilizan random_state=42 para reproducibilidad
4. **Visualizaciones:** Todas las visualizaciones utilizan matplotlib y seaborn para máxima claridad

---

🛠️ Requisitos
bashpip install pandas numpy matplotlib seaborn scikit-learn wooldridge

---

# Repositorio de Ejercicios de Python - Victor Guzman

![Notebook Check](https://github.com/TU_USUARIO/TU_REPO/workflows/📚%20Notebook%20Health%20Check/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 👤 Autor

**Victor Guzman**

---

## 📄 Licencia

Este repositorio contiene material educativo y ejercicios de práctica.

---

## 🤝 Contribuciones

Este es un repositorio de aprendizaje personal. Sin embargo, sugerencias y mejoras son bienvenidas.

---

## 📚 Referencias

- Dataset MROZ: Wooldridge package
- Encuesta Nacional de Hogares: Datos públicos de empleo
- Documentación oficial de Pandas, NumPy, Scikit-learn

---

**Última actualización:** 2024

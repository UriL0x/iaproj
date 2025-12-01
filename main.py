import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

''' Leer datos desde un archivo CSV '''
data = pd.read_csv('data.csv')

''' Explorar los datos '''
print(data.head())
print(data.describe())
print(data.info())
print(data.sample(n=5))
input("Presiona Enter para continuar...")
os.system('clear' if os.name == 'posix' else 'cls')

''' Preprocesar los datos '''
# Limpiar datos (eliminar filas con valores nulos)
data = data.dropna()

# Normalizar valores para que todos esten escritos con . en vez de ,
# En columna "Azucares"
data['Azucares'] = data['Azucares'].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
# En columna "Calorias"
data['Calorias'] = data['Calorias'].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
# En columna "Peso"
data['Peso'] = data['Peso'].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
# En columna "Diametro"
data['Diametro'] = data['Diametro'].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
# En columna "Altura"
data['Altura'] = data['Altura'].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

# Normalizar columna "Marca"  y convertir a numerico
data['Marca'] = data['Marca'].str.lower().str.strip()
data['Marca'] = data['Marca'].replace({'marmela': 'marinela'})  
marca_mapping = {'gamesa': 1, 'nabisco': 2, 'marinela': 3}
data['Marca'] = data['Marca'].map(marca_mapping)

print("Datos despues del preprocesamiento:")
print(data.sample(n=5))
input("Presiona Enter para continuar...")
os.system('clear' if os.name == 'posix' else 'cls')

''' Entrenamiento y evalucaion de modelos '''
X = data[['Azucares', 'Calorias', 'Peso', 'R', 'G', 'B', 'Chispas', 'Relleno']]
y = data[['Categoria', 'Marca']]

# KNN
from sklearn.neighbors import KNeighborsClassifier
print("K-Nearest Neighbors")

# Predecir la Marca
X_train, X_test, y_train, y_test = train_test_split(X, y['Marca'], test_size=0.2, random_state=42)
scaler = StandardScaler()   

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_accuracy_brand = knn.score(X_test, y_test)

print(f"KNN Accuracy[Marca]: {knn_accuracy_brand * 100:.2f}%")

# Predecir la Categoria
X_train, X_test, y_train, y_test = train_test_split(X, y['Categoria'], test_size=0.2, random_state=42)
scaler = StandardScaler()   

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_accuracy_category = knn.score(X_test, y_test)
print(f"KNN Accuracy[Categoria]: {knn_accuracy_category * 100:.2f}%")

# Teorema de bayes
from sklearn.naive_bayes import GaussianNB
print("Teorema de Bayes")

# Predecir la Marca
X_train, X_test, y_train, y_test = train_test_split(X, y['Marca'], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_accuracy_brand = gnb.score(X_test, y_test)
print(f"Naive Bayes Accuracy[Marca]: {gnb_accuracy_brand * 100:.2f}%")

# Predecir la Categoria
X_train, X_test, y_train, y_test = train_test_split(X, y['Categoria'], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_accuracy_category = gnb.score(X_test, y_test)
print(f"Naive Bayes Accuracy[Categoria]: {gnb_accuracy_category * 100:.2f}%")

# Maquina de vectores de soporte vectorial
from sklearn.svm import SVC
print("Support Vector Machine") 

# Predecir la Marca
X_train, X_test, y_train, y_test = train_test_split(X, y['Marca'], test_size=0.2, random_state=42)
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc_accuracy_brand = svc.score(X_test, y_test)
print(f"SVM Accuracy[Marca]: {svc_accuracy_brand * 100:.2f}%")

# Predecir la Categoria
X_train, X_test, y_train, y_test = train_test_split(X, y['Categoria'], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc_accuracy_category = svc.score(X_test, y_test)
print(f"SVM Accuracy[Categoria]: {svc_accuracy_category * 100:.2f}%")

# Graficos de barras para comparar los modelos
models = ['KNN', 'Naive Bayes', 'SVM']
marca_accuracies = [knn_accuracy_brand * 100, gnb_accuracy_brand * 100, svc_accuracy_brand * 100]
categoria_accuracies = [knn_accuracy_category * 100, gnb_accuracy_category * 100, svc_accuracy_category * 100]
x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, marca_accuracies, width, label='Marca')
bars2 = ax.bar(x + width/2, categoria_accuracies, width, label='Categoria')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Comparacion de modelos por accuracy')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(bars1)
autolabel(bars2)    
fig.tight_layout()
plt.show()
plt.savefig('model_comparison.png')




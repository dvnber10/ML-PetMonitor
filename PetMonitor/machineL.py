import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Cargar datos
data = pd.DataFrame({
    "Raza": ["Labrador", "Siamés", "Pastor Alemán"],
    "Actividad": [1, 3, 2],
    "Apetito": [2, 3, 1],
    "Sueño": [3, 1, 2],
    "Vocalizacion": [3, 1, 2],
    "Enfermedad": ["Hipotiroidismo", "Estrés", "Ansiedad"]
})

# Codificar raza y enfermedad
encoder = LabelEncoder()
data['Raza'] = encoder.fit_transform(data['Raza'])
data['Enfermedad'] = encoder.fit_transform(data['Enfermedad'])

# Separar características y etiquetas
X = data[['Raza', 'Actividad', 'Apetito', 'Sueño', 'Vocalizacion']]
y = data['Enfermedad']

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
predictions = model.predict(X_test)


# Evaluar modelo
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

with open("modelo_random_forest.pkl", "wb") as file:
    pickle.dump(model, file)

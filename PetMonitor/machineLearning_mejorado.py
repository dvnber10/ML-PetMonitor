import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Cargar datos de Excel
file_path = "registro_mascotas.csv"
df = pd.read_csv(file_path, sep=';')
print(df.head)

encoder = LabelEncoder()
Raza = LabelEncoder()
Enfermedad = LabelEncoder()
df['Raza'] = Raza.fit_transform(df['Raza'])
df['Enfermedad'] = Enfermedad.fit_transform(df['Enfermedad'])
df['Actividad'] = encoder.fit_transform(df['Actividad'])
df['Apetito'] = encoder.fit_transform(df['Apetito'])
df['Sueño'] = encoder.fit_transform(df['Sueño'])
df['Vocalización'] = encoder.fit_transform(df['Vocalización'])

X = df[['Raza', 'Actividad', 'Apetito', 'Sueño', 'Vocalización']]
y = df['Enfermedad']


# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#cargar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Hacer predicciones
predictions = model.predict(X_test)


# Evaluar modelo
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

with open("modelo_random_forest.pkl", "wb") as file:
    pickle.dump(model, file)

# Guardar el scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Guardar el LabelEncoder para 'Raza'
with open("raza_encoder.pkl", "wb") as file:
    pickle.dump(Raza, file)

# Guardar el LabelEncoder para 'Enfermedad'
with open("enfermedad_encoder.pkl", "wb") as file:
    pickle.dump(Enfermedad, file)

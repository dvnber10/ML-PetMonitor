from fastapi import FastAPI
import pickle
import numpy as np

# Crear instancia de FastAPI
app = FastAPI()

# Cargar el modelo y los encoders
with open("modelo_random_forest.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("raza_encoder.pkl", "rb") as file:
    raza_encoder = pickle.load(file)

with open("enfermedad_encoder.pkl", "rb") as file:
    enfermedad_encoder = pickle.load(file)

razas_originales = [str(raza) for raza in raza_encoder.classes_]
# Ruta para predicción

@app.post("/DetectarEnfermedad")
def predecir(raza: str, actividad: int, apetito: int, sueño: int, vocalizacion: int):
    # Validar la raza
    print(raza_encoder.classes_)
    if raza not in razas_originales:
        return {
             "error": f"La raza '{raza}' no está reconocida. Razas válidas: {', '.join(razas_originales)}"
        }
    
    # Codificar la raza y escalar los datos
    raza_codificada = raza_encoder.transform([raza])[0]
    input_data = np.array([[raza_codificada, actividad, apetito, sueño, vocalizacion]])
    input_data_scaled = scaler.transform(input_data)
    
    # Realizar predicción
    prediccion_codificada = model.predict(input_data_scaled)
    enfermedad_predicha = enfermedad_encoder.inverse_transform(prediccion_codificada)[0]
    
    return {"Enfermedad_Predicha": enfermedad_predicha}
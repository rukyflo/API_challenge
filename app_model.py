import os
import pickle
import subprocess

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


os.chdir(os.path.dirname(__file__))


app = Flask(__name__)
app.config["DEBUG"] = True


# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a la API del Team 3, equipo compuesto por: Rubén, Guillem, Genma, Bastien :)"


"""
La petición de prueba sería:
http://127.0.0.1:5000/api/v1/predict?income_cat=15&rooms_per_house=2&total_rooms=3&housing_median_age=8&bedrooms_ratio=2
"""

# Enruta la funcion al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods=["GET"])
def predict():  # Ligado al endpoint '/api/v1/predict', con el método GET
    model = pickle.load(open("ad_model.pkl", "rb"))
    '''
    ocean_proximity_ocean = request.args.get("ocean_proximity_<1H OCEAN", None)
    ocean_proximity_inland = request.args.get("ocean_proximity_INLAND", None)
    ocean_proximity_island = request.args.get("ocean_proximity_ISLAND", None)
    ocean_proximity_bay = request.args.get("ocean_proximity_NEAR BAY", None)
    ocean_proximity_near_ocean = request.args.get("ocean_proximity_NEAR OCEAN", None)
    '''
    income_cat = request.args.get("income_cat", None)
    rooms_per_house = request.args.get("rooms_per_house", None)
    total_rooms = request.args.get("total_rooms", None)
    housing_median_age = request.args.get("housing_median_age", None)
    bedrooms_ratio = request.args.get("bedrooms_ratio", None)

    list_features = [float(income_cat), float(rooms_per_house), float(total_rooms), float(housing_median_age), float(bedrooms_ratio)]
    
    print(type(model))

    for feature in list_features:
        print(f"La feature {feature} es del tipo {type(feature)}")

    for feature in list_features:
        if feature is None:
            return "Args empty, the data are not enough to predict, STUPID!!!!"
    else:
        prediction = model.predict([[float(income_cat), float(rooms_per_house), float(total_rooms), float(housing_median_age), float(bedrooms_ratio)]])
    return jsonify({"predictions": prediction[0]})


"""
La petición de prueba sería:
http://127.0.0.1:5000/api/v1/retrain
"""


@app.route("/api/v1/retrain", methods=["GET"])
# Enruta la funcion al endpoint /api/v1/retrain
def retrain():  # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists("data/ejemplo_housing.csv"):
        data = pd.read_csv("data/ejemplo_housing.csv")
        data = data.drop(columns=['ocean_proximity'])

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=["median_house_value"]), data["median_house_value"], test_size=0.20, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=["median_house_value"]), data["median_house_value"])
        pickle.dump(model, open("ad_model.pkl", "wb"))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"




@app.route("/webhook", methods=["POST"])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = "/home/rukyflo/myFlaskApp"
    servidor_web = "/var/www/rukyflo_pythonanywhere_com_wsgi.py"

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if "repository" in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload["repository"]["name"]
            clone_url = payload["repository"]["clone_url"]

            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify(
                    {"message": "El directorio del repositorio no existe!"}
                ), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(["git", "pull", clone_url], check=True)
                subprocess.run(
                    ["touch", servidor_web], check=True
                )  # Trick to automatically reload PythonAnywhere WebServer
                return jsonify(
                    {"message": f"Se realizó un git pull en el repositorio {repo_name}"}
                ), 200
            except subprocess.CalledProcessError:
                return jsonify(
                    {
                        "message": f"Error al realizar git pull en el repositorio {repo_name}"
                    }
                ), 500
        else:
            return jsonify(
                {
                    "message": "No se encontró información sobre el repositorio en la carga útil (payload)"
                }
            ), 400
    else:
        return jsonify({"message": "La solicitud no contiene datos JSON"}), 400


if __name__ == "__main__":
    app.run()
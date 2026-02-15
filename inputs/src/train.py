import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Criando dados manualmente
data = {
    "temperatura": [18, 20, 22, 25, 27, 30, 32, 35],
    "vendas": [100, 130, 150, 200, 240, 300, 330, 400]
}

df = pd.DataFrame(data)

X = df[["temperatura"]]
y = df["vendas"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.start_run()

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

mlflow.log_metric("mse", mse)
mlflow.sklearn.log_model(model, "modelo_sorvete")

mlflow.end_run()

print("Modelo treinado com sucesso!")

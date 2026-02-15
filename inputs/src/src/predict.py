import mlflow.sklearn
import pandas as pd

model = mlflow.sklearn.load_model("models:/modelo_sorvete/latest")

nova_temp = pd.DataFrame([[30]], columns=["temperatura"])

previsao = model.predict(nova_temp)

print(f"Previsão de vendas: {previsao[0]} sorvetes")

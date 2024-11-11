import torch
from ChessNetwork import ChessNetwork  # Reemplaza con la ruta correcta
import os

print(os.listdir())

# Asegúrate de que la ruta al archivo sea correcta
model_path = 'chess_model_iter_9.pt'
device = 'cpu'  # Cambia a 'cuda' si usas GPU

# try:
model = ChessNetwork(num_res_blocks=8, num_channels=128)
model.to(device)

# Carga los pesos del modelo
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print("Modelo cargado correctamente")
# except Exception as e:
#     print(f"Error al cargar el modelo: {e}")
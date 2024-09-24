import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions
    
# Función para crear secuencias de tiempo para LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length][0]  # Precio a predecir (se puede cambiar la columna objetivo)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Cargar datos
df = pd.read_csv('eq_fund.csv', delimiter=';')
df = df.replace(' ', 0)
df['abrir'] = df['abrir'].str.replace(',', '.').astype(float)
df['min'] = df['min'].str.replace(',', '.').astype(float)
df['max'] = df['max'].str.replace(',', '.').astype(float)
df.fillna(0, inplace=True)
data = df[['abrir', 'min', 'max']].values

# Escalar los datos entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Definición de la longitud de la secuencia
sequence_length = 300  # Longitud de la secuencia (se puede ajustar)
X, y = create_sequences(scaled_data, sequence_length)

# Verificar si la GPU está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando el dispositivo: {device}')

# Convertir a tensores de PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Crear un dataset y dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Inicializar el modelo
input_size = 3  # Tres variables de entrada (abrir, min, max)
hidden_layer_size = 128  # Más neuronas para mayor capacidad de representación
output_size = 1  # Predecir una variable (precio)

model = LSTMModel(input_size, hidden_layer_size, output_size).to(device)

# Definir función de pérdida y optimizador
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
epochs = 100
train_losses = []
for epoch in range(epochs):
    for seq, labels in dataloader:
        optimizer.zero_grad()
        y_pred = model(seq)
        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'modelo_lstm.pth')
print('Modelo guardado como modelo_lstm.pth')

# Predicciones
model.eval()

# Crear las predicciones para el conjunto completo
with torch.no_grad():
    predictions = model(X_tensor).cpu().numpy()

# Invertir la normalización para obtener los valores originales
predicted_prices = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, 0]
real_prices = scaler.inverse_transform(np.concatenate((y_tensor.cpu().numpy().reshape(-1, 1), np.zeros((y_tensor.shape[0], 2))), axis=1))[:, 0]

# Visualizar los resultados
plt.figure(figsize=(10,6))
plt.plot(real_prices, label='Precios Reales')
plt.plot(predicted_prices, label='Predicciones')
plt.legend()
plt.title('Comparación de Precios Reales y Predicciones')
plt.show()

# Clasificar para calcular métricas de evaluación
y_test_classes = np.where(real_prices > np.median(real_prices), 1, 0)  # Dividir en clases según la mediana
y_pred_classes = np.where(predicted_prices > np.median(predicted_prices), 1, 0)

# Calcular métricas
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes)
recall = recall_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Matriz de confusión
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Baja", "Sube"], yticklabels=["Baja", "Sube"])
plt.ylabel('Valores Reales')
plt.xlabel('Predicciones')
plt.title('Matriz de Confusión')
plt.show()

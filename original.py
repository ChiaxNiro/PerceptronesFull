import numpy as np

# Funciones para generación de datos
def generate_linear_data(num_samples):
    X = []
    for _ in range(num_samples):
        x_vals = np.linspace(0, 10, 10)
        y_vals = 2 * x_vals + np.random.randn(10) * 0.5  # Línea con ruido
        X.append(np.column_stack((x_vals, y_vals)))
    return np.array(X)

def generate_circular_data(num_samples):
    X = []
    for _ in range(num_samples):
        angles = np.linspace(0, 2 * np.pi, 10)
        x_vals = 5 * np.cos(angles) + np.random.randn(10) * 0.1
        y_vals = 5 * np.sin(angles) + np.random.randn(10) * 0.1
        X.append(np.column_stack((x_vals, y_vals)))
    return np.array(X)

def generate_random_data(num_samples):
    X = []
    for _ in range(num_samples):
        x_vals = np.random.randn(10) * 10
        y_vals = np.random.randn(10) * 10
        X.append(np.column_stack((x_vals, y_vals)))
    return np.array(X)

# Función para aplanar los datos de 10x2 a 20x1
def flatten_data(X):
    return X.reshape(X.shape[0], -1).T  # Convierte cada ejemplo de 10x2 en un vector de 20x1

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

# Clase Perceptron con capa de entrada, oculta y salida
class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01  # Pesos de entrada a capa oculta
        self.b1 = np.zeros((hidden_size, 1))  # Bias de capa oculta
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01  # Pesos de capa oculta a salida
        self.b2 = np.zeros((output_size, 1))  # Bias de capa de salida
    
    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = sigmoid(Z1)  # Activación Sigmoid en capa oculta
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2)  # Activación Softmax en capa de salida
        return A1, A2

# Función de pérdida (Cross-Entropy)
def cross_entropy_loss(Y, A2):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A2 + 1e-8)) / m
    return loss

# Retropropagación (Backpropagation)
def backward(X, A1, A2, Y, W2, W1, b2, b1, learning_rate=0.01):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    return W1, b1, W2, b2

# Función de entrenamiento
def train_perceptron(perceptron, X_train, Y_train, epochs=1000, learning_rate=0.01):
    losses = []
    for epoch in range(epochs):
        A1, A2 = perceptron.forward(X_train)
        loss = cross_entropy_loss(Y_train, A2)
        losses.append(loss)
        perceptron.W1, perceptron.b1, perceptron.W2, perceptron.b2 = backward(
            X_train, A1, A2, Y_train, 
            perceptron.W2, perceptron.W1, 
            perceptron.b2, perceptron.b1, 
            learning_rate=learning_rate
        )
        if epoch % 100 == 0:
            print(f"Época {epoch}, Pérdida: {loss}")
    return losses

# Función de predicción
def predict(perceptron, X):
    _, A2 = perceptron.forward(X)
    predictions = np.argmax(A2, axis=0)
    return predictions

# Creación de etiquetas en formato one-hot
def create_labels(num_samples, class_index):
    labels = np.zeros((3, num_samples))
    labels[class_index, :] = 1
    return labels

# Generación de datos de entrenamiento
linear_data = generate_linear_data(100)
circular_data = generate_circular_data(100)
random_data = generate_random_data(100)

# Preparación de los datos de entrenamiento
X_train = np.concatenate([
    flatten_data(linear_data),
    flatten_data(circular_data),
    flatten_data(random_data)
], axis=1)

Y_train = np.concatenate([
    create_labels(100, 0),  # Movimiento lineal
    create_labels(100, 1),  # Movimiento circular
    create_labels(100, 2)   # Movimiento aleatorio
], axis=1)

# Inicializar y entrenar el perceptrón
perceptron = Perceptron(input_size=20, hidden_size=5, output_size=3)
train_perceptron(perceptron, X_train, Y_train, epochs=1000, learning_rate=0.01)

# Generación de datos de prueba
linear_data_test = generate_linear_data(30)
circular_data_test = generate_circular_data(30)
random_data_test = generate_random_data(30)

# Preparación de los datos de prueba
X_test = np.concatenate([
    flatten_data(linear_data_test),
    flatten_data(circular_data_test),
    flatten_data(random_data_test)
], axis=1)

Y_test = np.concatenate([
    create_labels(30, 0),  # Movimiento lineal
    create_labels(30, 1),  # Movimiento circular
    create_labels(30, 2)   # Movimiento aleatorio
], axis=1)

# Evaluación del modelo
predictions = predict(perceptron, X_test)
true_labels = np.argmax(Y_test, axis=0)
accuracy = np.mean(predictions == true_labels) * 100
print(f"Precisión total en el conjunto de prueba: {accuracy:.2f}%")
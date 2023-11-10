import numpy as np

# Función de activación (step function)
def step_function(x):
    return 1 if x >= 0 else 0

# Función de entrenamiento del perceptrón
def train_perceptron(inputs, outputs, learning_rate, epochs):
    # Inicialización de pesos y sesgo de manera aleatoria
    num_inputs = len(inputs[0])
    weights = np.random.rand(num_inputs)
    bias = np.random.rand()

    for _ in range(epochs):
        error_sum = 0
        for i in range(len(inputs)):
            input_data = inputs[i]
            output = outputs[i]
            prediction = step_function(np.dot(input_data, weights) + bias)
            error = output - prediction
            error_sum += error

            # Actualización de pesos y sesgo
            weights += learning_rate * error * input_data
            bias += learning_rate * error

        if error_sum == 0:
            break

    return weights, bias

# Compuerta lógica OR
inputs_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_or = np.array([0, 1, 1, 1])

learning_rate = 0.1
epochs = 100
weights_or, bias_or = train_perceptron(inputs_or, outputs_or, learning_rate, epochs)

# Compuerta lógica AND
inputs_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_and = np.array([0, 0, 0, 1])

weights_and, bias_and = train_perceptron(inputs_and, outputs_and, learning_rate, epochs)

# Prueba del perceptrón entrenado
def test_perceptron(inputs, weights, bias):
    results = []
    for input_data in inputs:
        prediction = step_function(np.dot(input_data, weights) + bias)
        results.append(prediction)
    return results

# Prueba de las compuertas lógicas OR y AND
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

print("OR Gate Output:")
print(test_perceptron(test_inputs, weights_or, bias_or))

print("AND Gate Output:")
print(test_perceptron(test_inputs, weights_and, bias_and))
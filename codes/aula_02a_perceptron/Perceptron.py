import numpy as np

class Perceptron:
    def __init__(self, learn_rate=0.3, num_epochs=4):
        self.__learn_rate = learn_rate
        self.__num_epochs = num_epochs

    def train(self, inputs, targets):
        #vetor de pesos; inicia-se pesos iguais a 0, de tamanho de acordo com o número de atributos (colunas) + 1 (bias)
        self.__weights  = np.zeros(inputs.shape[1]+1)

        for epoch in range(self.__num_epochs):
            print(f'Epoch {epoch}: [', end='')
            for x, target in zip(inputs, targets):
                print('=', end='')
                error = self.__learn_rate * (target - self.predict(x))
                self.__weights[1:] += error * x
                self.__weights[0] += error
            print(']', end='\n\n')

    def predict(self, set):
        v = np.dot(set, self.__weights[1:]) + self.__weights[0]
        return np.where(v >= 0.0, 1, -1)
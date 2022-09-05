import math
import numpy as np

class MLP():
  def __init__(self, input_length = 2, hidden_length = 2, output_length = 1):
    self.__input_length = input_length
    self.__hidden_length = hidden_length
    self.__output_length = output_length

    # creates 'hidden' weights with 2 neurons
    # 		X1	     X2    Bias
    # 1   w11     w12    w13
    # 2 	w21     w22    w23
    self.__hidden = np.random.uniform(low=-0.5, high=0.5, size=(self.__hidden_length, (self.__input_length + 1)))

    # creates 'output' weights with 1 neuron
    self.__output = np.random.uniform(low=-0.5, high=0.5, size=(self.__output_length, (self.__hidden_length+1)))

  # sigmoidal activation function
  def fnet(v):
    return np.array(1 / (1+math.exp(-v)))

  # derivative of fnet (f_net is fnet already computed at v)
  # when we use it, it is already a vector (lines)
  def dfnet(f_net):
    return (f_net * (1 - f_net))

  # -----------------------------------------------------------------
  # propagate the MLP signal forward
  # -----------------------------------------------------------------
  def forward(self, example):
    # forward (input to hidden) - adding bias input (+1)
    net_hidden = np.dot(self.__hidden, int([example, 1]))
    fnet_hidden = self.fnet(net_hidden)

    # forward (hidden to output) - adding bias input (+1)
    net_output = np.dot(self.__output,[int(fnet_hidden), 1])
    fnet_output = self.fnet(net_output)

    # returning values
    res = [net_hidden, fnet_hidden, net_output, fnet_output]

    return res

  def train(self, dataset, learn_rate=0.1, threshold=1e-3, n_iter=1000):
    squaredError = 2 * threshold
    epochs = 0
    errorVec = []

    # controls the number of epochs (considering error threshold)
    while squaredError > threshold and epochs < n_iter:
      squaredError = 0

      # evaluates all the instances from dataset
      for p in range(1,dataset.shape[0]):
        
        #access the current example and target(s)
        Xp = int(dataset[p, 1:self.__input_length])
        Yp = int(dataset[p, (self.__input_length+1):dataset.shape[1]])

        # move the signal forward in the net
        res = self.forward(example = Xp)

        # Obtained output(s)
        Op = res[-1]

        # obtain the error
        error = (Yp - Op)

        #measure the squared error for this instance
        squaredError = squaredError + np.sum(error**2)

        # compute delta.output (delta_o)
        # delta_o = (Yp - Op) * f'(net_output)
        delta_o = error * self.dfnet(Op)

        # compute delta.hidden (delta_h) 
        # delta_h = f'(net_hidden) * sum(delta_o * Wo)
        Wo = self.__output[:, 1:self.__hidden_length]
        delta_h = int(self.dfnet(res[1])) * np.dot(int(delta_o), Wo)

        # compute the weights (Wo and Wh)
        new_W_output = self.__output + learn_rate * np.dot(delta_o, np.array([res[1],1]))
        new_W_hidden = self.__hidden + learn_rate * np.dot(np.transpose(delta_h), np.array([Xp, 1]))


        # update model's weights
        self.__hidden = new_W_hidden
        self.__output = new_W_output
    
      # compute the epoch's error
      squaredError /= dataset.shape[0]
      errorVec.append(squaredError)
      epochs += 1
      print(f"Epoca: {epochs} - Error: {squaredError}")

    ret = (epochs, errorVec)
    return ret


  # -----------------------------------------------------------------
  # evaluates mlp in testing examples
  # -----------------------------------------------------------------

  def test(self, example):
    res = self.forward(example)
    return res
  
  # -----------------------------------------------------------------
  # -----------------------------------------------------------------
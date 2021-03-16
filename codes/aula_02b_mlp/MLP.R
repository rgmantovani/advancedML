# -----------------------------------------------------------------
# Activation function and its derivative
# -----------------------------------------------------------------

# sigmoidal activation function
fnet = function(v){
  return (1 / (1+exp(-v)))
}

# derivative of fnet (f_net is fnet already computed at v)
# when we use it, it is already a vector (lines)
dfnet = function(f_net){
  return (f_net * (1 - f_net))
} 

# -----------------------------------------------------------------
#  Initiates a model
# -----------------------------------------------------------------

mlp.create = function(input.length = 2, hidden.length = 2,
  output.length = 1) {

  # create a new object (list)
  model = list()
  
  # mlp topology
  model$input.length  = input.length
  model$hidden.length = hidden.length
  model$output.length = output.length

  # functions
  model$fnet  = fnet
  model$dfnet = dfnet
  
  # creates 'hidden' weights (wh) with 2 neurons
  # 		X1	     X2    Bias
  # 1   w11     w12    w13
  # 2 	w21     w22    w23
  wh = runif(min = -0.5, max = 0.5, n = hidden.length * (input.length + 1))
  model$hidden = matrix(data = wh, nrow = hidden.length, ncol = input.length + 1)

  # creates 'output' weights (wo) with 1 neuron
  wo = runif(min = -0.5, max = 0.5, n = output.length * (hidden.length + 1))
  model$output = matrix(data = wo, nrow = output.length, ncol = hidden.length + 1)

  return(model)
}

# -----------------------------------------------------------------
# propagate the MLP signal forward
# -----------------------------------------------------------------

mlp.forward = function(model, example) {

  # forward (input to hidden) - adding bias input (+1)
  net.hidden  = model$hidden %*% as.numeric(c(example, 1))
  fnet.hidden = model$fnet(net.hidden)

  # forward (hidden to output) - adding bias input (+1)
  net.output = model$output %*% c(as.numeric(fnet.hidden),1)
  fnet.output = model$fnet(net.output)

  # returning values
  res = list(net.hidden = net.hidden, fnet.hidden = fnet.hidden,
    net.output = net.output, fnet.output = fnet.output)

  return(res)
}

# -----------------------------------------------------------------
# train a mlp
# -----------------------------------------------------------------

mlp.train = function(model, dataset, lrn.rate = 0.1, threshold = 1e-3, 
   n.iter = 1000) {
  
  squaredError = 2 * threshold
  epochs = 0
  errorVec = c()
  
  # controls the number of epochs (considering error threshold)
  while(squaredError > threshold & epochs < n.iter) {

    squaredError = 0
    
    # evaluates all the instances from dataset
    for(p in 1:nrow(dataset)) {
      
      #access the current example and target(s)
      Xp = as.numeric(dataset[p,1:model$input.length])
      Yp = as.numeric(dataset[p,(model$input.length+1):ncol(dataset)])
      
      # move ths ginal forward in the net
      res = mlp.forward(model = model, example = Xp)
      
      # Obtained output(s)
      Op  = res$fnet.output
      
      # obtain the error
      error = (Yp - Op)
      
      #measure the squared error for this instance
      squaredError = squaredError + sum(error^2)
      
      # compute delta.output (delta_o)
      # delta_o = (Yp - Op) * f'(net_output)
      delta.output = error * model$dfnet(Op)
    
      # compute delta.hidden (delta_h) 
      # delta_h = f'(net_hidden) * sum(delta_o * Wo)
      Wo = model$output[, 1:model$hidden.length]
      delta.hidden = as.numeric(model$dfnet(res$fnet.hidden)) * (as.numeric(delta.output) %*% Wo)
      
      # compute the weights (Wo and Wh)
      new.W.output = model$output + lrn.rate * (delta.output %*% as.vector(c(res$fnet.hidden,1)))
      new.W.hidden = model$hidden + lrn.rate * (t(delta.hidden) %*% as.vector(c(Xp,1)))
      
      # update model's weights
      model$hidden = new.W.hidden
      model$output = new.W.output
    }
    
    # compute the epoch's error
    squaredError = squaredError / nrow(dataset)
    errorVec = c(errorVec, squaredError)
    epochs = epochs + 1
    cat("Epoca: ", epochs, "- Error: ", squaredError, "\n")
  }
  
  # returning the obtained model
  ret = list(model = model, epochs = epochs, errorVec = errorVec)
  return(ret)
}

# -----------------------------------------------------------------
# evaluates mlp in testing examples
# -----------------------------------------------------------------
mlp.test = function(model, example) {
  res = mlp.forward(model = model, example = example)
  return(res)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

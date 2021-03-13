# -----------------------------------------------------------------
#  creates a model - OK
# -----------------------------------------------------------------

mlp.create = function(input.length = 2, hidden.length = 2,
  output.length = 1, fnet = fnet, dfnet = dfnet) {

  # create a new object (list)
  model = list()
  model$input.length  = input.length
  model$hidden.length = hidden.length
  model$output.length = output.length

  # creates 'hidden' weights (wh)
  wh = runif(min = -0.5, max = 0.5, n = hidden.length * (input.length + 1))
  model$hidden = matrix(data = wh, nrow = hidden.length, ncol = input.length + 1)

  # creates 'output' weights (wo)
  wo = runif(min = -0.5, max = 0.5, n = output.length * (hidden.length + 1))
  model$output = matrix(data = wo, nrow = output.length, ncol = hidden.length + 1)

  # activation functions
  model$fnet  = fnet
  model$dfnet = dfnet

  return(model)
}

# -----------------------------------------------------------------
# OK
# -----------------------------------------------------------------

mlp.forward = function(model, example) {

  # adding bias input (+1)
  example = as.numeric(c(example, 1))

  # forward (input to hidden)
  net.hidden  = model$hidden %*% example
  fnet.hidden = model$fnet(net.hidden)

  # forward (hidden to output)
  net.output = model$output %*% c(as.numeric(fnet.hidden),1)
  fnet.output = model$fnet(net.output)

  #returning values
  res = list(net.hidden = net.hidden, fnet.hidden = fnet.hidden,
    net.output = net.output, fnet.output = fnet.output)

  return(res)
}

# -----------------------------------------------------------------
# train a mlp
# -----------------------------------------------------------------

mlp.train = function(model, dataset, lrn.rate = 0.1,
  threshold = 1e-3, n.iter = 1000) {

  squaredError = 2 * threshold
  epochs = 0
  errorVec = c()

  # controls the number of epochs (considering error threshold)
  while(squaredError > threshold) {

    if(epochs > n.iter) {
      break
    }

    squaredError = 0

    # evaluates all the instances from dataset
    for(p in 1:nrow(dataset)) {

      #acess the current example and target(s)
      Xp = as.numeric(dataset[p,1:model$input.length])
      Yp = as.numeric(dataset[p,(model$input.length+1):ncol(dataset)])

      # move forward in the net
      res = mlp.forward(model = model, example = Xp)
      Op  = res$fnet.output

      # obtain the error
      error = (Yp - Op)
      squaredError = squaredError + sum(error^2)

      # compute delta.output
      # delta_o = (Yp - Op) * f'(net_output)
      delta.output = error * model$dfnet(res$fnet.output)

      # compute delta.hidden
      # delta_h = f'(net_hidden) * sum(delta_o * wo)
      wo = model$output[, 1:model$hidden.length]
      delta.hidden = as.numeric(model$dfnet(res$fnet.hidden)) *
        (as.numeric(delta.output) %*% wo)

      # update weights (Wo and Wh)
      new.W.output = model$output + lrn.rate *
        (delta.output %*% as.vector(c(res$fnet.hidden,1)))
			new.W.hidden = model$hidden + lrn.rate * (t(delta.hidden) %*%
        as.vector(c(Xp,1)))

      model$hidden = new.W.hidden
      model$output = new.W.output

  	}

    # compute the epoch's error
    squaredError = squaredError / nrow(dataset)
    errorVec = c(errorVec, squaredError)
    epochs = epochs + 1
    cat("Epoca: ", epochs, "- Error: ", squaredError, "\n")

  }

  ret = list(model = model, epochs = epochs, errorVec = errorVec)
  return(ret)
}

# -----------------------------------------------------------------
# evaluates mlp in a training example
# -----------------------------------------------------------------

mlp.test = function(model, example) {

  res = mlp.forward(model = model, example = example)
  return(res)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

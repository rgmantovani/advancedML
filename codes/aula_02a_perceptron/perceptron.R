# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# Funções: 
# - perceptron.train: treina um perceptron simples no conjunto de treinamento fornecido como entrada
# - perceptron.test: faz a predição para os valores contidos em um conjunto de teste

# -------------------------------------------------------------------------------------------------
# Perceptron.train 
#    - train.set: conjunto de treinamento
#    - weights: são os pesos sinápticos da rede
#    - lrn.rate: taxa de aprendizado, para ajuste sináptico
#    - epochs: numero maximo de iterações para treinamento (caso não chaja convergência)
# -------------------------------------------------------------------------------------------------
perceptron.train = function(train.set, weights = NULL, lrn.rate = 0.3, n.iter = 1000) {
  
  epochs = 0
  error  = TRUE
  
  if(is.null(weights)) {
    weights = runif(ncol(train.set)-1,-1,1)
  }
  
  cat("Initial weights: ", weights, "\n")
  avgErrorVec = c()
  class.id = ncol(train.set)
  
  # while there is an error in training examples, and epochs < n.iter
  while(error & epochs < n.iter) {
    
    # limiting the number of epochs
    # if(epochs > n.iter) {
    # break
    # }
    
    error  = FALSE
    epochs = epochs + 1
    avgError = 0
    cat("Epoca:", epochs,"\n")
    
    for(i in 1:nrow(train.set)) {
      
      example  = as.numeric(train.set[i,])
      
      # spike
      x = example[-class.id]
      v = as.numeric(x %*% weights)
      
      # output
      # it could also be:
      y = ifelse(v >=0, +1, -1) # y = sign(v)
      
      avgError = avgError + ((example[class.id] - y)^2)
      
      # updating weights (only to misclassified patterns)
      if(example[class.id] != y) {
        cat(" - adjustment required ...\n")
        weights = weights + lrn.rate * (example[class.id] - y) * example[-class.id]
        error = TRUE
      }
      print(weights)
    }
    
    avgError = avgError/nrow(train.set)
    avgErrorVec = c(avgErrorVec, avgError)
    cat("Epoch: ", epochs," - Avg Error = ", avgError, "\n")
  }
  
  # returning object with some slots
  obj = list(weights = weights, avgErrorVec = avgErrorVec, epochs = epochs)
  
  cat("\n* Finished after: ",epochs,"  epochs\n")
  return(obj)
}

# -----------------------------------------------------------------
# Perceptron.test: avalia novos exemplos depois que o modelo foi treinado, 
# usando os pesos sinápticos objetidos no treinamento
#    - conjunto de teste
#    - weights: são os pesos sinápticos obtidos no treinamento
# -----------------------------------------------------------------

perceptron.predict = function(test.set, weights) {
  
  v = as.numeric(test.set %*% weights)
  # it also works as: v = sum(test.set * weights)
  y = ifelse(v>=0, +1, -1)
  return(y)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------
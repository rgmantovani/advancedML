# -----------------------------------------------------------------
# -----------------------------------------------------------------

source("./MLP.R")

# -----------------------------------------------------------------
# -----------------------------------------------------------------

x1    = c(0,0,1,1)
x2    = c(0,1,0,1)
class = c(0,1,1,0)
dataset = data.frame(x1, x2, class)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

#  activation function
fnet = function(x) {
  y = 1 /(1 + exp(-x))
  return(y)
}

# derivative function
dfnet = function(x) {
  y = x * (1-x)
  return(y)
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------

set.seed(1)

# modelo = mlp.architecture(activation.f = fnet, d_activation = dfnet)
# obj = mlp.backpropagation(modelo, dataset, eta=0.1, threshold=1e-3)
# pred = mlp.forward(obj$modelo, c(0,0))

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# instantiate a mlp network
model = mlp.create(input.length = 2, hidden.length = 2,
  output.length = 1, fnet = fnet, dfnet = dfnet)

# model$hidden = modelo$hidden
# model$output = modelo$output

obj2 = mlp.train(model = model, dataset = dataset, lrn.rate = 0.1,
  threshold = 1e-2, n.iter = 100000)

pred2 = mlp.test(model = obj2$model, example = c(0,0))

# predicted value output
round(pred2$fnet.output)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

library(ggplot2)

df = data.frame(1:length(obj2$errorVec), obj2$errorVec)
colnames(df) = c("epochs", "error")
g = ggplot(df, aes(x = epochs, y = error)) + geom_line()

# -----------------------------------------------------------------
# -----------------------------------------------------------------

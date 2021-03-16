# -----------------------------------------------------------------
# -----------------------------------------------------------------

# loading our mlp implementation
source("./MLP.R")
set.seed(1)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

#creates XOR dataset
x1    = c(0,0,1,1)
x2    = c(0,1,0,1)
class = c(0,1,1,0)
dataset = data.frame(x1, x2, class)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# instantiate a mlp network
model = mlp.create(input.length = 2, hidden.length = 2,
  output.length = 1)

# defines the stopping criteria with and error < 0.001 and epochs < 100k
obj = mlp.train(model = model, dataset = dataset, lrn.rate = 0.1,
  threshold = 1e-2, n.iter = 100000)

# predicting the output of an example
pred2 = mlp.test(model = obj$model, example = c(0,0))
print(round(pred2$fnet.output))

# -----------------------------------------------------------------
# Plotting the convergence curve
# -----------------------------------------------------------------

library(ggplot2)

df = data.frame(1:length(obj$errorVec), obj$errorVec)
colnames(df) = c("epochs", "error")
g = ggplot(df, aes(x = epochs, y = error)) + geom_line()
g

# -----------------------------------------------------------------
# -----------------------------------------------------------------

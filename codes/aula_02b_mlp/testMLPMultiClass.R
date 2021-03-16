# -----------------------------------------------------------------
# -----------------------------------------------------------------

# loading our mlp implementation
source("./MLP.R")
set.seed(24)

# -----------------------------------------------------------------
# handling iris dataset
# -----------------------------------------------------------------

data  = iris

# split class into 3 target variables
data$class.Setosa     = 0
data$class.Versicolor = 0
data$class.Virginica  = 0

# renaming factors to numeric values
data$class.Setosa[which(data$Species == "setosa")] = 1
data$class.Versicolor[which(data$Species == "versicolor")] = 1
data$class.Virginica[which(data$Species == "virginica")] = 1

# removes the old class column
data$Species = NULL

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# split data into training and testing folds (holdout)
train.data = data[c( 1:30,  51:80, 101:130), ]
test.data  = data[c(31:50, 81:100, 131:150), ]

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# defining a network with (4,3,3) neurons
model = mlp.create(input.length = 4, hidden.length = 3,
  output.length = 3)

# training the MLP model
obj = mlp.train(model = model, dataset = train.data, lrn.rate = 0.1,
    threshold = 1e-2, n.iter = 10000)

# testing each example from the testing set
aux = lapply(1:nrow(test.data), function(i) {
  
  # for each example, it gets its prediction 
  pred = mlp.test(model = obj$model, example = test.data[i, 1:model$input.length])
 
  # returns the obtained output for each example
  ret = as.numeric(round(pred$fnet.output))
  return(ret)
})

# getting the real class values
y.real = test.data[, 5:7]

# getting obtained predictions
y.pred = data.frame(do.call("rbind", aux))
colnames(y.pred) = colnames(y.real)

# checking predictions vs expected classes
acc = 0
for(i in 1:nrow(test.data)){
  # if it is right, increments counter
  if(which.max(y.pred[i,]) == which.max(y.real[i,])) {
    acc = acc + 1
  }
}

# accuracy = number of corrected classification / number of examples
res = acc/nrow(test.data)
print(res)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

library("ggplot2")

# Plotting the convergence error
df = data.frame(1:length(obj$errorVec), obj$errorVec)
colnames(df) = c("epochs", "error")
g = ggplot(df, aes(x = epochs, y = error)) + geom_line()
g

# -----------------------------------------------------------------
# -----------------------------------------------------------------
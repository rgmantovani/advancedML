# -----------------------------------------------------------------
# -----------------------------------------------------------------

# loading our perceptron implementation
source("./perceptron.R")
library("ggplot2")

set.seed(24)

# -----------------------------------------------------------------
# Getting iris dataset
# -----------------------------------------------------------------

data = iris

# Statistics
dim(data)
names(data)
summary(data$Species)

# -----------------------------------------------------------------
# plot: data separability
# -----------------------------------------------------------------

#jpeg('data_separability.jpg', width = 8, height = 8, units = 'in', res = 300)
pairs(data[,-ncol(data)], col=data$Species)
#dev.off()

# -----------------------------------------------------------------
# subsetting dataset
# -----------------------------------------------------------------

# Selecting only two features + Class (Species)
X = data[ , c(2,4,5)]

# Binarizing the problem = 2 classes (Setosa VS Virginica + Versicolor)
X$Species = ifelse(X$Species == "setosa", +1, -1)
X = cbind(1, X)
colnames(X)[1] = "bias"

# -----------------------------------------------------------------
# training perceptron
# -----------------------------------------------------------------

obj = perceptron.train(train.set = X, lrn.rate = 0.1)

# -----------------------------------------------------------------
# Plot: error convergence
# -----------------------------------------------------------------

df = data.frame(1:obj$epochs, obj$avgErrorVec)
colnames(df) = c("epoch", "avgError")

g = ggplot(df, mapping = aes(x = epoch, y = avgError))
g = g + geom_line() + geom_point()
g = g + scale_x_continuous(limit = c(1, nrow(df)))
print(g)

# -----------------------------------------------------------------
# plot: viweing the data
# -----------------------------------------------------------------

X$Species = as.factor(X$Species)
g2 = ggplot(X, mapping = aes(x = Sepal.Width, y = Petal.Width,
                             colour = Species, shape = Species))
g2 = g2 + geom_point(size = 3) + theme_bw()
g2 

# -----------------------------------------------------------------
# plot: the obtained hyperplane
# -----------------------------------------------------------------

w0 = obj$weights[1] # bias weight
w1 = obj$weights[2]
w2 = obj$weights[3]

slope     = -(w0/w2)/(w0/w1)
intercept = -w0/w2

g3 = g2 + geom_abline(intercept = intercept, slope = slope)
g3
#ggsave(g3, file = paste0("dataset_hyperplane_",
#  seed.value,".jpg"), width = 6, height = 6, dpi = 480)

# -----------------------------------------------------------------
# testing data
# -----------------------------------------------------------------

test1 = c(1,2.5,0.3)
res1 = perceptron.predict(test.set = test1, weights = obj$weights)
print(test1)
print(res1)

test2 = c(1,2.5,0.5)
res2 = perceptron.predict(test.set = test2, weights = obj$weights)
print(test2)
print(res2)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
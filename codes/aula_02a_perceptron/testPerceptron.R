# -----------------------------------------------------------------
# -----------------------------------------------------------------

# loading our perceptron implementation
source("./perceptron.R")
library("ggplot2")

# -----------------------------------------------------------------
# -----------------------------------------------------------------

set.seed(42)

# -----------------------------------------------------------------
# loading datasets
# -----------------------------------------------------------------

dataset = read.csv(file = "./datasets/and.csv")
# dataset = read.csv(file = "./datasets/or.csv")
# dataset = read.csv(file = "./datasets/artificial.csv")
# dataset = read.csv(file = "./datasets/xor.csv")

# -----------------------------------------------------------------
#  training perceptron
# -----------------------------------------------------------------

obj = perceptron.train(train.set = dataset, lrn.rate = 0.5)

# -----------------------------------------------------------------
# plotting training error convergence
# -----------------------------------------------------------------

df = data.frame(1:obj$epochs, obj$avgErrorVec)
colnames(df) = c("epoch", "avgError")

# Avg training error
g = ggplot(df, mapping = aes(x = epoch, y = avgError))
g = g + geom_line() + geom_point()
g = g + scale_x_continuous(limit = c(1, nrow(df)))
g
#ggsave(g, file = paste0("perceptron_convergence_",
#  seed.value,".jpg"), width = 7.95, height = 3.02, dpi = 480)

# -----------------------------------------------------------------
# plot: the obtained hyperplane
# -----------------------------------------------------------------

dataset$D = as.factor(dataset$D)
g2 = ggplot(dataset, mapping = aes(x = X1, y = X2, colour = D, shape = D))
g2 = g2 + geom_point(size = 3) + theme_bw()
g2 

# hyperplane
w0 = obj$weights[1] # bias weight
w1 = obj$weights[2]
w2 = obj$weights[3]

slope     = -(w0/w2)/(w0/w1)
intercept = -w0/w2

g2 = g2 + geom_abline(intercept = intercept, slope = slope)
g2

#ggsave(g2, file = paste0("perceptron_hyperplane_",
#  seed.value,".jpg"), width = 6, height = 6, dpi = 480)

# -----------------------------------------------------------------
# testing data
# -----------------------------------------------------------------

test1 = c(1,2,2)
res1 = perceptron.predict(test.set = test1, weights = obj$weights)
print(test1)
print(res1)

test2 = c(1,4,4)
res2 = perceptron.predict(test.set = test2, weights = obj$weights)
print(test2)
print(res2)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
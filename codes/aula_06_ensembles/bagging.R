# ----------------------------------------------------------
# data : training set
# n.learners: number of learners to be used in the committe
# ----------------------------------------------------------

bagging.train = function(data, n.learners, subagging = FALSE) {

    model = list()
    aux = lapply(1:n.learners, function(i) {

      # subsampling
      ids = NULL
      if(!subagging) {
        ids = sample(x = 1:nrow(data), size = nrow(data), replace = TRUE)
      } else {
        #subagging: sampling with half of the samples, and without replacement
        ids = sample(x = 1:nrow(data), size = nrow(data)/2, replace = FALSE)
      }
      sub = data[ids, ]

      #fitting a tree ("[[class]] ~ . ")
      fstr = paste0(colnames(data)[ncol(data)], " ~ .")
      tree = rpart::rpart(formula = as.formula(fstr), data = sub)

      #returning the fitted model
      return(tree)
    })

    # returning a list of models (committe)
    model$committe   = aux
    model$n.learners = n.learners

    return(model)
}

# ----------------------------------------------------------
# model: fitted model
# data : testing data, withtout class column
# ----------------------------------------------------------
bagging.test = function(model, data) {

  # generating predictions
  aux.com = lapply(1:model$n.learners, function(k){
    preds = predict(object = model$committe[[k]], data)
    preds = round(preds)

    tree.pred = lapply(1:nrow(preds), function(i) {
      return(names(which.max(preds[i,])))
    })
    tree.pred = unlist(tree.pred)
    return(tree.pred)
  })

  # committe agreement
  votes = do.call("rbind", aux.com)

  aux.votes = lapply(1:ncol(votes), function(n) {
    # return the most predicted class
    tab.votes = table(votes[,n])
    max.id = which.max(as.numeric(tab.votes))
    return(names(tab.votes)[max.id])
  })

  # the committe agreement
  com.agreement = unlist(aux.votes)
  return(com.agreement)
}

# ----------------------------------------------------------
# testing functions
# ----------------------------------------------------------

# using iris as an example
train = iris[ c(1:30, 51:80, 101:131), ]
test  = iris[-c(1:30, 51:80, 101:131), ]
realClass = test$Species


# ----------------
# bagging
# ----------------

# training model
model = bagging.train(data = train, n.learners = 10)
# obtaining prediction
preds = bagging.test(model = model, data = test[,-ncol(test)])
preds = factor(preds, levels = levels(realClass))

# computing accuracy
tab.res = table(realClass,preds)
acc = sum(diag(tab.res))/sum(tab.res)
print(acc)


# ----------------
# subagging
# ----------------

# training subagging model
model2 = bagging.train(data = train, n.learners = 10, subagging = TRUE)
# obtaining prediction
preds2 = bagging.test(model = model2, data = test[,-ncol(test)])
preds2 = factor(preds2, levels = levels(realClass))

# computing accuracy
tab.res2= table(realClass,preds2)
acc2 = sum(diag(tab.res2))/sum(tab.res2)
print(acc2)

# ----------------------------------------------------------
# ----------------------------------------------------------

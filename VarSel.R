
# Perform cross-validation with embedded variable selection

library(mlr)

gam.data$Date <- NULL


# define classification task
task <- makeClassifTask(data = gam.data,
                        target = "Acc", positive = "TRUE")

configureMlr(on.learner.error = "warn", on.error.dump = TRUE)

# define a MLR learner
mL <- makeLearner("classif.binomial",
                  link = "logit",
                  predict.type = "prob",
                  fix.factors.prediction = TRUE)


# define the resampling task (RepCV oder CV?)
resampling <- makeResampleDesc(method = "RepCV", folds = 5, reps = 50)


# choice of feature selection method (with sequential search forward)
ctrl <- makeFeatSelControlSequential(method = "sfs", maxit = NA, alpha = 0.001)


# Perform the method
result <- selectFeatures(mL,task, resampling, control = ctrl, measures = mlr::auc)

# Retrieve AUC and selected variables
analyzeFeatSelResult(result)

lrn <- makeFeatSelWrapper(mL, resampling, control = ctrl, show.info = FALSE)


set.seed(12345)
cv_glm <- mlr::resample(lrn, task, resampling, extract = getFeatSelResult, measures = mlr::auc, models = TRUE)

summary(cv_glm$measures.test$auc)




# train the defined classifier on the predefined task
model <- train(lrn, task)

# extract the model outcomes
model_glm <- getLearnerModel(model)
summary(model_glm)

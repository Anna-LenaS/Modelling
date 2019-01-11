#   Description:
#   This script aims at performing cross-validation with embedded stepwise variable selection 
#   and assessment of the performance of the resulting model (selected features)

library(mlr)

### Nested resampling

# modify data
gam.data$Date <- NULL

# define classification task
task <- makeClassifTask(data = gam.data,
                        target = "Acc", positive = "TRUE")

configureMlr(on.learner.error = "warn", on.error.dump = FALSE)

# define a MLR learner
lrn <- makeLearner("classif.binomial",
                   link = "logit",
                   predict.type = "prob",
                   fix.factors.prediction = TRUE)

# define the resampling task 
# RepCV: here iters = folds * reps
# later: increase reps = 2 to reps = 100
inner <- makeResampleDesc(method = "RepCV", folds = 5, reps = 2, predict = "both")

# choice of feature selection method (with sequential search forward)
ctrl <- makeFeatSelControlSequential(method = "sfs", maxit = NA, alpha = 0.01)
ctrl

# generate wrapped learner
wrapper_glm <- makeFeatSelWrapper(lrn, resampling = inner, control = ctrl, show.info = TRUE)

# train the defined classifier on the predefined task
mod <- train(wrapper_glm, task)
mod
# extract result of the feature selection 
sfeats = getFeatSelResult(mod)
sfeats
# the selected features are:
sfeats$x
# extract the model outcomes
model_glm <- getLearnerModel(mod)
summary(model_glm)


### Outer resampling loop

# later: increase reps = 2 to reps = 100
outer = makeResampleDesc(method = "RepCV", folds = 5, reps = 2, predict = "both")

# Parrelization of ML models
library(parallelMap) 
parallelStart(mode = "socket", cpus = 3, level = "mlr.selectFeatures")

set.seed(12345, kind = "L'Ecuyer-CMRG")

# The 5-fold cross-validated performance of the learner can be computed as follows:
res <- resample(wrapper_glm, task, extract = getFeatSelResult,
                models = TRUE,
                resampling = outer,
                show.info = TRUE, measures = list(auc, timetrain))

parallelStop()

res
res$aggr
summary(res$measures.test$auc)
res$measures.test

# extract the selected feature sets in the individual resampling iterations 
lapply(res$models, getFeatSelResult)

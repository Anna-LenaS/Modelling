library(mlr)

gam.data$Date <- NULL

# define classification task
task <- makeClassifTask(data = gam.data,
                        target = "Acc", positive = "TRUE",
                        coordinates = ???)

configureMlr(on.learner.error = "warn", on.error.dump = TRUE)

# define a MLR learner
lrn <- makeLearner("classif.binomial",
                   link = "logit",
                   predict.type = "prob",
                   fix.factors.prediction = TRUE)

# define the resampling task (RepCV oder CV?)
# Patrick: RepCV for a robust result -> But I assume that your data is 
# spatial right? So then please use "SpRepCV". You need to supply coordinates
# for it to work. See https://mlr.mlr-org.com/articles/tutorial/handling_of_spatial_data.html
resampling <- makeResampleDesc(method = "SpRepCV", folds = 5, reps = 2)

# choice of feature selection method (with sequential search forward)
# Patrick: Just some questions: Why is alpha set to 0.001?
ctrl <- makeFeatSelControlSequential(method = "sfs", maxit = NA, alpha = 0.001)

# Perform the method
# result <- selectFeatures(mL,task, resampling, control = ctrl, measures = mlr::auc)

# Retrieve AUC and selected variables
analyzeFeatSelResult(result)

wrapper_glm <- makeFeatSelWrapper(mL, resampling, control = ctrl, show.info = T)

# parallelize your code
# "multicore" only works on Linux. If you need to run on Windows, use "socket" instead
# and remove "mc.set.seed"
parallelStart(
  mode = "multicore", cpus = 3, level = "mlr.selectFeatures",
  mc.set.seed = TRUE
)
set.seed(12345, kind = "L'Ecuyer-CMRG")

res <- resample (wrapper_glm, task, extract = getFeatSelResult,
                 models = TRUE,
                 resampling = resampling,
                 show.info = TRUE, measures = list(auc, timetrain)
)

parallelStop()

summary(res$measures.test$auc)

# train the defined classifier on the predefined task
# Patrick: Here you also need to make a SFS on the task. In the same way as you 
# did it in each fold of the CV
model <- train(lrn, task)

# extract the model outcomes
model_glm <- getLearnerModel(model)
summary(model_glm)

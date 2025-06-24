library(readr)
library(dplyr)
library(fastDummies)
library(lightgbm)
library(pROC)
library(ggplot2)


# 1. Lectura de datos
train_values <- read_csv("data_old/train_values.csv")
train_labels <- read_csv("data_old/train_labels.csv")
test_values <- read_csv("data_old/test_values.csv")
test_labels <- read_csv("data_old/test_labels.csv")

# 2. Preprocessing
# Convertir a matriz

X_train <- as.matrix(
  train_values |> 
  select(-patient_id) |>
  mutate(thal = as.factor(thal)) |>
  dummy_cols(select_columns = "thal", remove_selected_columns = TRUE)
)
y_train <- train_labels |> select(heart_disease_present) |> as.matrix()

X_test  <- as.matrix(
  test_values|> 
  select(-patient_id) |>
    mutate(thal = as.factor(thal)) |>
    dummy_cols(select_columns = "thal", remove_selected_columns = TRUE)
  )
y_test  <- test_labels |> select(heart_disease_present) |> as.matrix()

# Crear datasets para LightGBM
dtrain <- lgb.Dataset(
  data = X_train, 
  label = y_train,
  )
dtest  <- lgb.Dataset(
  data = X_test, 
  label = y_test,
  )

# 3. Definr LightGBM
params <- list(
  objective = "binary",
  metric = "auc",
  boosting = "gbdt",
  verbosity = 3,
  num_leaves = 31,
  learning_rate = 0.05,
  bagging_fraction = 0.8,
  feature_fraction = 0.9,
  bagging_freq = 1
)

# 4. Entrenar modelo
model <- lgb.train(
  params = params,
  nrounds = 100,
  data = dtrain,
  valids = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  record = TRUE
)

# 5. Predicciones
# Predicciones sobre el set de test
pred_probs <- predict(model, X_test)

# 6. Gini plot
auc_train <- model$record_evals$train$auc$eval
auc_test <- model$record_evals$test$auc$eval
gini_train <- 2 * unlist(auc_train) - 1
gini_test <- 2 * unlist(auc_test) - 1

gini_df <- data.frame(
  iter = seq_along(gini_train),
  train = gini_train,
  test = gini_test
)

ggplot(gini_df, aes(x = iter)) +
  geom_line(aes(y = train, color = "Train"), size = 1.2) +
  geom_line(aes(y = test, color = "Test"), size = 1.2) +
  scale_color_manual(values = c("Train" = "#1f77b4", "Test" = "#ff7f0e")) +
  labs(title = "Evolucion del Gini por ronda",
       x = "Iteracion",
       y = "Gini",
       color = "Conjunto")+
  theme_minimal()

# 7. ROC plot
roc_train <- roc(y_train, predict(model, X_train))
roc_test  <- roc(y_test, predict(model, X_test))

df_train <- data.frame(
  tpr = rev(roc_train$sensitivities),
  fpr = rev(1 - roc_train$specificities),
  dataset = "Train"
)

df_test <- data.frame(
  tpr = rev(roc_test$sensitivities),
  fpr = rev(1 - roc_test$specificities),
  dataset = "Test"
)

roc_df <- rbind(df_train, df_test)

ggplot(roc_df, aes(x = fpr, y = tpr, color = dataset)) +
  geom_line(size = 1.2) +
  scale_color_manual(values = c("Train" = "#1f77b4", "Test" = "#ff7f0e")) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = "Curva ROC",
    x = "1 - Especificidad",
    y = "Sensibilidad"
  )+
  theme_minimal()


roc_curve <- roc(y_test, pred_probs)
auc(roc_curve)  # Área bajo la curva
plot(roc_curve)

lgb.importance(model)
lgb.importance(model) |> 
  lgb.plot.importance(top_n = 20)

lgb.tree(model = model, tree_index = 1)

model$best_iter
model$record_evals

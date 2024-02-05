library("dplyr")
library(ggplot2)
library(patchwork)
library(gplots)
library(glmnet)
library(caret)
library(pROC)

data<-read.csv("train.csv")
str(data)
dim(data)
names<-colnames(data)
names
#data<-na.omit(data)
sum(is.na(data))

#removing unnamed..0,track_id,artists,album_name,track_name columns
df<-data[,-c(1,2,3,4,5)]
colnames(df)
str(df)

#drop_genres <- c('acoustic','alternative','afrobeat','alt-rock','ambient','anime','black-metal','bluegrass','blues','brazil','breakbeat','british','cantopop','chicago-house','children','chill','club','comedy','dancehall','death-metal','deep-house','detroit-techno','disney','drum-and-bass','dub','dubstep','edm','electro','emo','folk','forro','french','funk','garage','german','gospel','goth','grindcore','groove','grunge','guitar','happy','hard-rock','hardcore','hardstyle','heavy-metal','honky-tonk','idm','indian','indie-pop','indie','industrial','iranian','j-dance','j-idol','j-pop','j-rock','k-pop','kids','latin','latino','malay','mandopop','metalcore','minimal-techno','mpb','new-age','opera','pagode','party','piano','pop-film','power-pop','progressive-house','psych-rock','punk-rock','punk','reggae','reggaeton','rock-n-roll','rockabilly','romance','sad','salsa','samba','sertanejo','show-tunes','singer-songwriter','ska','sleep','songwriter','soul','spanish','study','swedish','synth-pop','tango','techno','trance','trip-hop','turkish','world-music')
genres<-c('classical','country','electronic','hip-hop','jazz','rock','pop','rock','blues','reggae')

# Remove specified genres
#df_filtered <- df[!df$track_genre %in% drop_genres, ]
df_filtered<-df[df$track_genre %in% genres,]
X<-model.matrix(track_genre~.,data=df_filtered)
y<-as.factor(df_filtered$track_genre)

df_filtered$track_genre <- as.factor(df_filtered$track_genre)
#df_filtered$mode<-as.factor(df_filtered$mode)
#df_filtered$explicit<-as.factor(df_filtered$explicit)

numeric_df <- df_filtered[, sapply(df_filtered, is.numeric)]

#df_filtered <- df[!df$track_genre %in% drop_genres, ]
#X<-model.matrix(track_genre~.,data=df_filtered)
#y<-as.factor(df$track_genre)


# Plot pairs
#pairs(numeric_df)
table(y)
#genres<-unique(y)
#genres

cor_matrix<-cor(numeric_df)
cor_matrix

variable_names <- colnames(cor_matrix)

heatmap.2(cor_matrix, 
          trace = "none", # Turn off row/column dendrogram
          col = colorRampPalette(c("black", "white", "red"))(20),
          main = "Correlation Matrix",
          dendrogram="none",
          #xlab = "Variables", ylab = "Variables",
          key = TRUE, # Add color key for the gradient
          key.title = NA, # Remove the default key title
          key.xlab = "Correlation", # Add x-axis label for the key
          key.ylab = NULL, # Remove the default y-axis label for the key
          Rowv = variable_names,  # Set row order
          Colv = variable_names,  # Set column order
          density.info = "none", # Turn off density plot
          symkey = FALSE, # Do not plot a symmetric key
          add.expr = {
            # Add text annotations for correlation values
            for (i in 1:nrow(cor_matrix)) {
              for (j in 1:ncol(cor_matrix)) {
                text(i ,15- j , format(cor_matrix[i, j], digits = 2),
                     col = "black", cex = 0.8)
              }
            }
          },
          #width = 6, # Adjust width of the plot
          #height = 6, # Adjust height of the plot
          cexRow = 0.8, # Adjust text size for rows
          cexCol = 0.8 # Adjust text size for columns
          
)
library(igraph)

S <- var(numeric_df)
R <- -cov2cor(solve(S))

thr <- 0.3

G <- abs(R)>thr
diag(G) <- 0

#Gi <- as(G, "igraph")
#plot(Gi)
Gi <- graph_from_adjacency_matrix(as.matrix(G), mode = "undirected", weighted = NULL, diag = FALSE)
plot(Gi,vertex.color="white")
#tkplot(Gi, vertex.color="white")



#EDA

#library(ggplot2)


# Extract numeric variables
numeric_vars <- c("popularity", "duration_ms", "danceability", "energy", "key", "loudness",
                   "speechiness", "acousticness", "instrumentalness", "liveness",
                  "valence", "tempo")

#par(mfrow=c(1,3))
# Loop through each numeric variable
for (var in numeric_vars) {
  bin_width <- diff(range(df_filtered[[var]])) / 50 
  # Histogram
  hist_plot <- ggplot(df_filtered, aes(x = !!sym(var))) +
    geom_histogram(binwidth = bin_width, position = "identity", alpha = 0.7) +
    labs(title = paste("Histogram of", var))
  #print(hist_plot)
  # Density distribution
  density_plot <- ggplot(df_filtered, aes(x = !!sym(var), fill = track_genre)) +
    geom_density(alpha = 0.7) +
    labs(title = paste("Density Distribution of", var))
  #print(density_plot)
  # Box plot
  box_plot <- ggplot(df_filtered, aes(x = track_genre, y = !!sym(var), fill = track_genre)) +
    geom_boxplot() +
    labs(title = paste("Box Plot of", var))
  #print(box_plot)
  # Combine subplots for each variable
  subplot = hist_plot / density_plot / box_plot
  
  # Display the combined subplots
  print(subplot)
}
#par(mfrow=c(1,1))













set.seed(1)
train <- sample(1:nrow(X), 4*nrow(X)/5)
test <- (-train)
#X.train<-X[train,]
#X.test<-X[test,]
y.train<-y[train]
y.test <- y[test]

library(doParallel)
library(foreach)
# Register parallel backend
cl <- makeCluster(detectCores())
registerDoParallel(cl)

X_scaled <- scale(X)[,-1]
X_scaled.train<-X_scaled[train,]
X_scaled.test<-X_scaled[test,]

fit_ridge<-glmnet(X_scaled[train,],y[train],family="multinomial",alpha=0)
plot(fit_ridge)
lambda_interval<-10^seq(2,-8,length=100)
cvfit_ridge<-cv.glmnet(X_scaled[train,],y[train],family="multinomial",alpha=0,lambda=lambda_interval,parallel=TRUE)
plot(cvfit_ridge)
#glm with Lambda equal 0,
pred <- predict(fit_ridge, s = 0, newx = X_scaled[test, ],
                      exact = TRUE, type="class",x = X_scaled[train, ], y = y[train])
pred_accuracy<-mean(pred==y.test)
pred_accuracy

bestlam <- cvfit_ridge$lambda.min
bestlam

ridge_pred=predict(cvfit_ridge,newx=X_scaled[test,],type="class", s="lambda.min")
ridge_pred_accuracy<-mean(ridge_pred==y.test)
ridge_pred_accuracy

# identify the lambda.1se 
bestlam.1se <- cvfit_ridge$lambda.1se
bestlam.1se

ridge_1se_pred=predict(cvfit_ridge,newx=X_scaled[test,],type="class", s="lambda.1se")
ridge_1se_pred_accuracy<-mean(ridge_pred==y.test)
ridge_1se_pred_accuracy

conf_matrix <- table(Actual = y.test, Predicted = ridge_1se_pred)
print(conf_matrix)

metrics <- confusionMatrix(conf_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]

# Create a data frame with the metrics
metrics_df <- data.frame(
  Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
)

# Print or use the metrics_df data frame
print(metrics_df)


#Lasso

fit_lasso<-glmnet(X_scaled[train,],y[train],family="multinomial",alpha=1)

plot(fit_lasso,xvar="lambda")


custom_lambda_values <- 10^seq(10, -4, length = 20)  # Customize the range and number of lambda values

cvfit_lasso <- cv.glmnet(X_scaled[train,], y[train], family = "multinomial", alpha = 1, parallel = TRUE)#, nlambda = length(custom_lambda_values), lambda = custom_lambda_values, maxit = 200000, solver = "l-bfgs")


#plot(cvfit_lasso)
bestlam_lasso <- cvfit_lasso$lambda.min
bestlam_lasso

bestlam.1se <- cvfit_lasso$lambda.1se
bestlam.1se

abline(v=log(bestlam_lasso),col="red")
abline(v=log(bestlam.1se),col="green")



plot(cvfit_lasso)


str(coef(fit_lasso,s=bestlam.1se))
lasso_coefficients<-coef(fit_lasso,s=bestlam.1se)
lasso_coefficients
non_zero_counts <- sapply(lasso_coefficients, function(mat) sum(nnzero(mat)))

# Print the non-zero counts
print(non_zero_counts)
#best_lasso<-glmnet(X_scaled[train,],y[train],family="multinomial",alpha=1,lambda=bestlam.1se,maxit=200000)
#best_lasso
#coef(cvfit_lasso,s=bestlam.1se)

#
# Number of bootstrap samples
#num_bootstraps <- 1000

# Function to extract coefficients from glmnet object
#get_coefs <- function(model, indices) {
#  coef(model, s = cv.glmnet(X_scaled.train[indices, ], y.train[indices], family="multinomial",alpha = 1)$lambda.1se)
#}

# Bootstrap procedure
#boot_coefs <- replicate(num_bootstraps, get_coefs(cvfit_lasso, sample(1:nrow(X_scaled.train), nrow(X_scaled.train), replace = TRUE)))


num_cores <- detectCores()
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Define the number of bootstraps
num_bootstraps <- 100

# Define the function to get coefficients from bootstrapped samples
get_coefs <- function( indices) {
  #coef(model, s = cv.glmnet(X_scaled.train[indices, ], y.train[indices], family = "multinomial", alpha = 1)$lambda.1se)
  library(glmnet)
  model <- glmnet(
    x = X_scaled.train[indices, ],
    y = y.train[indices],
    family = "multinomial",
    alpha=1,
    parallel = TRUE
  )
  
  # Extract and return the coefficients as a matrix
  #return(as.matrix(coef(model,s=bestlam.1se)))
  coefs_list <- coef(model, s = bestlam.1se)
  
  # Convert the list of coefficient vectors to a list of matrices
  coefs_matrices <- lapply(coefs_list, as.matrix)
  
  return(coefs_matrices)
  }

# Bootstrap procedure in parallel
boot_coefs <- foreach(i = 1:num_bootstraps, .combine = "rbind") %dopar% {
  set.seed(i)  # Set a seed for reproducibility
  indices <- sample(1:nrow(X_scaled.train), nrow(X_scaled.train), replace = TRUE)
  get_coefs( indices)
}

# Stop the parallel backend
stopCluster(cl)

# Print or use the boot_coefs matrix
print(boot_coefs)

#boot_coefs_combined <- do.call(rbind, boot_coefs)

par(mfrow = c(3,3))

for (i in (1:ncol(boot_coefs))) {
  class_coefs <- boot_coefs[,i]
  class_coefs_c<-do.call(cbind,class_coefs)
  boxplot(t(class_coefs_c), main = colnames(boot_coefs)[i],
          col = "lightblue",
          names = colnames(class_coefs),xaxt="n")
  abline(h=0,col="red",lty=2)
  axis(1, at = seq_along(rownames(class_coefs_c)), labels = rownames(class_coefs_c), las = 2)
}
par(mfrow=c(1,1))

# Reset the plotting layout
par(mfrow = c(1, 1))
# Calculate standard errors
sd_lasso_coefs <- apply(boot_coefs, 1, sd)

# Calculate t-values
t_values_lasso <- coef(best_lasso) / se_coefs

# Print t-values
print(t_values)


df <- num_bootstraps  # Degrees of freedom

non_zero_indices <- which(coef(best_lasso) != 0)

p_values_ridge <- 2 * (1 - pt(abs(t_value[!is.na(t_values_lasso)]), df))


significant_p_values_lasso <- p_values_lasso <= 0.05

# Identify non-zero coefficients
lasso_coef <- predict(cvfit_lasso,type="coefficients",s=bestlam.1se)
head(lasso_coef)


# Create a data frame for non-zero coefficients
result_lasso_df <- data.frame(
  #Coefficient = rownames(coef(best_ridge))[non_zero_indices],
  Value = coef(best_lasso)[non_zero_indices],
  Standard_deviation = sd_lasso_coefs[non_zero_indices],
  T_Value = t_values_lasso[non_zero_indices],
  P_Value = p_values_lasso[non_zero_indices],
  P_value_significance = significant_p_values_lasso[non_zero_indices]
)



#fit_lasso<-glmnet(X[train,],y[train],family="multinomial",alpha=1)
#summary(fit_lasso)

lasso_pred<-predict(cvfit_lasso,newx=X_scaled[test,],type="class", s="lambda.min")
lasso_pred_accuracy<-mean(lasso_pred==y.test)
lasso_pred_accuracy

lasso_coef <- predict(cvfit_lasso,type="coefficients",s=bestlam_lasso)
lasso_coef
#lasso_coef[lasso_coef!=0]

# identify the lambda.1se 


lasso_1se_coef <- predict(cvfit_lasso,type="coefficients",s=bestlam.1se)
lasso_1se_coef
#lasso_1se_coef[lasso_1se_coef!=0]


lasso_1se_pred=predict(cvfit_lasso,newx=X_scaled[test,],type="class", s="lambda.1se")
lasso_1se_pred_accuracy<-mean(lasso_pred==y.test)
lasso_1se_pred_accuracy
conf_matrix <- table(Actual = y.test, Predicted = lasso_pred)
print(conf_matrix)

metrics <- confusionMatrix(conf_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]

# Create a data frame with the metrics
metrics_df <- data.frame(
  Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
)

# Print or use the metrics_df data frame
print(metrics_df)



#elastic net

set.seed(1)
# Initialize empty lists
alpha_list <- numeric()
error_min_list <- numeric()
error_1se_list <- numeric()
lambda_min_list <- numeric()
lambda_1se_list <- numeric()

#custom_lambda_sequence <- 10^seq(1, -6, length = 100)
for (a in seq(0, 1, by = 0.1)) {
  
  cvfit_elastic_net <- cv.glmnet(X_scaled[train,],y[train],family="multinomial", alpha = a)#,lambda=custom_lambda_sequence)
  #plot(cvfit_elastic_net)
  title(main = bquote(alpha == .(a)))
  # Extract alpha, lambda.min, lambda.1se, and cross-validated errors
  alpha_value <- cvfit_elastic_net$glmnet.fit$alpha
  lambda_min <- cvfit_elastic_net$lambda.min
  lambda_1se <- cvfit_elastic_net$lambda.1se
  error_min <- cvfit_elastic_net$cvm[cvfit_elastic_net$lambda == lambda_min]
  error_1se <- cvfit_elastic_net$cvm[cvfit_elastic_net$lambda == lambda_1se]
  
  # Append values to lists
  alpha_list <- c(alpha_list, a)
  error_min_list <- c(error_min_list, error_min)
  error_1se_list <- c(error_1se_list, error_1se)
  lambda_min_list <- c(lambda_min_list, lambda_min)
  lambda_1se_list <- c(lambda_1se_list, lambda_1se)
}

# Create a dataframe from the lists
results_df <- data.frame(
  alpha = seq(0, 1, by = 0.1),
  error_min = error_min_list,
  error_1se = error_1se_list,
  lambda_min = lambda_min_list,
  lambda_1se = lambda_1se_list
)

# Print the dataframe
print(results_df)

plot(results_df$alpha, results_df$error_min,ylim = c(min(results_df$error_min)-0.2, max(results_df$error_1se) + 0.1),
     xlab = "Alpha", ylab = "Minimum deviance", main = "Min binomial deviance vs. Alpha",
     pch = 16, col = "red", cex = 1.5)

# Add error bars (standard deviation)
arrows(results_df$alpha, results_df$error_min - (results_df$error_1se-results_df$error_min), results_df$alpha, results_df$error_1se,
       angle = 90, code = 3, length = 0.05, col = "black")

# Optionally, add points to highlight individual data points
points(results_df$alpha[which.min(results_df$error_min)], results_df$error_min[which.min(results_df$error_min)], pch = 16, col = "blue", cex = 1.5)


alpha_chosen<-results_df$alpha[which.min(results_df$error_min)]

best_lam_elastic_net<-results_df$lambda_1se[which.min(results_df$error_min)]

best_elastic_net<-glmnet(X_scaled[train,],y[train],family="multinomial",alpha=alpha_chosen,lambda=results_df$lambda_1se[which.min(results_df$error_min)],standardize = FALSE,lambda.factor=0.0000001)
#best_elastic_net
non_zero_coefficients_elastic_net<-coef(best_elastic_net)[coef(best_elastic_net)!=0]
non_zero_coefficients_elastic_net
length(non_zero_coefficients_elastic_net)


registerDoParallel(cores = 8)

get_coefs <- function(alpha_value, indices) {
  if (alpha_value == 1) {
    # Use best lambda for lasso
    lambda_value <- bestlam_lasso.1se
  } else if (alpha_value == 0) {
    # Use best lambda for ridge
    lambda_value <- bestlam_ridge
  } else if(alpha_value>0 & alpha_value<1) {
    lambda_value<-best_lam_elastic_net
  }else {
    stop("Invalid alpha_value. It should be between 0 and 1.")
  }
  
  # Perform the glmnet fitting with the selected lambda
  model <- glmnet(
    x = train_data[indices, ],
    y = train_response[indices],
    family = "multinomial",
    alpha = alpha_value,
    lambda = lambda_value,
    parallel = TRUE
  )
  
  # Extract and return the coefficients as a matrix
  return(as.matrix(coef(model)))
}

get_bootstrap_sample <- function(alph) {
  library(glmnet)
  get_coefs(alph, sample(1:nrow(train_data), nrow(train_data), replace = TRUE))
}



elastic_net_coefs_matrix <- foreach(i = 1:num_bootstraps, .combine = 'cbind') %dopar% {#%dopar% indicates that loop should be done in parallel
  get_bootstrap_sample(alpha_chosen)
}
stopImplicitCluster()

#save(list=c("elastic_net_coefs_matrix"),file="elastic_net_coefs_matrix.RData")
#load("elastic_net_coefs_matrix.RData")

# Calculate standard errors
#se_elastic_net_coefs <- apply(elastic_net_coefs_matrix, 1, function(x) sd(x) / sqrt(num_bootstraps))

sd_elastic_net_coefs<- apply(elastic_net_coefs_matrix, 1, function(x) sd(x))

# Calculate t-values
t_values_elastic_net <- coef(best_elastic_net) / (sd_elastic_net_coefs+1e-7)

t_values_elastic_net[coef(best_elastic_net)!=0]

# Calculate relative p-values
df <- num_bootstraps  # Degrees of freedom

non_zero_indices_en <- which(coef(best_elastic_net) != 0)

p_values_elastic_net <- 2 * (1 - pt(abs(t_values_elastic_net[non_zero_indices_en]), df))


significant_p_values_en <- p_values_elastic_net <= 0.05

# Identify non-zero coefficients
elastic_net_coef <- predict(best_elastic_net,type="coefficients")
#lasso_1se_coef
elastic_net_coef[elastic_net_coef!=0]

# Create a data frame for non-zero coefficients
result_elastic_net_df <- data.frame(
  #Coefficient = rownames(elastic_net_coef)[non_zero_indices_en],
  Value = coef(best_elastic_net)[non_zero_indices_en],
  Standard_deviation = sd_elastic_net_coefs[non_zero_indices_en],
  T_Value = t_values_elastic_net[non_zero_indices_en],
  P_Value = p_values_elastic_net,
  P_value_significance=significant_p_values_en
)

elastic_net_pred<-predict(best_elastic_net,newx=test_data,type="class")
elastic_net_pred_accuracy<-mean(elastic_net_pred==test_response)
elastic_net_pred_accuracy

conf_matrix <- table(Actual = test_response, Predicted = elastic_net_pred)
print(conf_matrix)


metrics <- confusionMatrix(conf_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$overall["Accuracy"]

# Create a data frame with the metrics
metrics_df <- data.frame(
  Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity,
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)






#KNN
library(class)


data.train <- data.frame(X_scaled.train, y.train)

# Define the training control with 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Specify the tuning grid for k (e.g., values from 1 to 10)
grid <- expand.grid(k = 1:10)

# Train the KNN model with cross-validation
set.seed(123)  # Set a seed for reproducibility
knn_model <- train(y.train ~ ., data = data.train, method = "knn", trControl = ctrl, tuneGrid = grid)

# Print the results and optimal k
print(knn_model)

# Optimal k value
optimal_k <- knn_model$bestTune$k
print(paste("Optimal k:", optimal_k))







k_value <- optimal_k  # Set the number of neighbors
knn_model <- knn(train = X_scaled[train,], test = X_scaled[test,], cl = df_filtered[train,16], k = k_value)

# Evaluate the model
conf_matrix <- table(Actual = y.test, Predicted = knn_model)
print(conf_matrix)
mean(knn_model==y.test)
#

metrics <- confusionMatrix(conf_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)







#install.packages("nnet")
library(nnet)

#Multinomial model

multinom_model <- multinom(track_genre ~ ., data = df_filtered[train,])
summary(multinom_model)
# Make Predictions
predictions <- predict(multinom_model, newdata = df_filtered[test,], type = "class")

# Evaluate the Model
confusion_matrix <- table(predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Display results
print("Confusion Matrix:")
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)


######
#Backward AIC
library(MASS)  # For stepAIC function

# Fit multinomial model using nnet package
model <- multinom(track_genre ~ ., data = df_filtered[train,])

# Perform stepwise selection using stepAIC function
step_model <- step(model, direction = "backward")
step_model$coefnames
#formula(step_model)
#attr(formula(step_model), "predvars")

step_model$anova

#step_model<-model <- multinom(track_genre ~ .-key, data = df_filtered[train,])
predictions <- predict(step_model, newdata = df_filtered[test,], type = "class")

# Evaluate the Model
confusion_matrix <- table(predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Display results
print("Confusion Matrix:")
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)

###########Backward BIC


model <- multinom(track_genre ~ ., data = df_filtered[train,])

# Perform stepwise selection using stepAIC function
step_model <- step(model, direction = "backward",k=log(length(train)))
step_model$coefnames

#formula(step_model)
#attr(formula(step_model), "predvars")
length(step_model$coefnames)
step_model$anova

predictions <- predict(step_model, newdata = df_filtered[test,], type = "class")

# Evaluate the Model
confusion_matrix <- table(predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Display results
print("Confusion Matrix:")
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)





#Forward feature selection(BIC)
selected_features <- c()
min_BIC <- Inf

while (TRUE) {
  remaining_features <- setdiff(colnames(df_filtered), c("track_genre", selected_features))
  
  if (length(remaining_features) == 0) {
    break
  }
  
  best_BIC <- Inf
  best_feature <- NULL
  
  for (feature in remaining_features) {
    current_model <- multinom(track_genre ~ ., data = df_filtered[train, c(selected_features, feature, "track_genre")])
    current_BIC <- BIC(current_model)
    
    if (current_BIC < best_BIC) {
      best_BIC <- current_BIC
      best_feature <- feature
    }
  }
  
  if (best_BIC < min_BIC) {
    min_BIC <- best_BIC
    selected_features <- c(selected_features, best_feature)
  } else {
    break  # No improvement in BIC, stop the loop
  }
}


# Display the selected features
print(selected_features)
length(selected_features)



multinom_model_bic <- multinom(track_genre ~ ., data = df_filtered[train,c(selected_features,"track_genre")])

# Make Predictions
predictions <- predict(multinom_model_bic, newdata = df_filtered[test,c(selected_features,"track_genre")], type = "class")

# Evaluate the Model
confusion_matrix <- table(predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Display results
print("Confusion Matrix:")
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)

#Forward feature selection(AIC)
selected_features <- c()
min_AIC <- Inf

while (TRUE) {
  remaining_features <- setdiff(colnames(df_filtered), c("track_genre", selected_features))
  
  if (length(remaining_features) == 0) {
    break
  }
  
  best_AIC <- Inf
  best_feature <- NULL
  
  for (feature in remaining_features) {
    current_model <- multinom(track_genre ~ ., data = df_filtered[train, c(selected_features, feature, "track_genre")])
    current_AIC <- AIC(current_model)
    
    if (current_AIC < best_AIC) {
      best_AIC <- current_AIC
      best_feature <- feature
    }
  }
  
  if (best_AIC < min_AIC) {
    min_AIC <- best_AIC
    selected_features <- c(selected_features, best_feature)
  } else {
    break  # No improvement in BIC, stop the loop
  }
}


# Display the selected features
print(selected_features)
length(selected_features)



multinom_model_aic <- multinom(track_genre ~ ., data = df_filtered[train,c(selected_features,"track_genre")])

# Make Predictions
predictions <- predict(multinom_model_aic, newdata = df_filtered[test,c(selected_features,"track_genre")], type = "class")

# Evaluate the Model
confusion_matrix <- table(predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Display results
print("Confusion Matrix:")
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)


#Backward feature selection(BIC)

selected_features <- colnames(df_filtered)[-which(colnames(df_filtered) == "track_genre")]  # start with all features
num_features <- ncol(df_filtered) - 1  # excluding the target variable
min_BIC <- Inf

for (i in 1:num_features) {
  # Loop through selected features
  for (feature in selected_features) {
    # Train model without the current feature
    formula <- reformulate(setdiff(selected_features, feature), response = "track_genre")
    

    current_model <- multinom(formula, data = df_filtered[train, ])
    # Calculate BIC for the model
    current_BIC <- BIC(current_model)
    
    # Print debugging information
    print(paste("Selected Features:", toString(setdiff(selected_features, feature))))
    print(paste("Excluded Feature:", feature))
    print(paste("Current BIC:", current_BIC))
    
    # Update selected features if BIC is reduced
    if (current_BIC < min_BIC) {
      min_BIC <- current_BIC
      selected_features <- setdiff(selected_features, feature)
    }
  }
}

# Display the selected features
print(selected_features)
length(selected_features)



multinom_model <- multinom(track_genre ~ ., data = df_filtered[train,c(selected_features,"track_genre")])

# Make Predictions
predictions <- predict(multinom_model, newdata = df_filtered[test,c(selected_features,"track_genre")], type = "class")

# Evaluate the Model
confusion_matrix <- table(predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Display results
print("Confusion Matrix:")
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)


#Backward feature selection(AIC)

selected_features <- colnames(df_filtered)[-which(colnames(df_filtered) == "track_genre")]  # start with all features
num_features <- ncol(df_filtered) - 1  # excluding the target variable
min_BIC <- Inf

for (i in 1:num_features) {
  # Loop through selected features
  for (feature in selected_features) {
    # Train model without the current feature
    formula <- reformulate(setdiff(selected_features, feature), response = "track_genre")
    
    
    current_model <- multinom(formula, data = df_filtered[train, ])
    # Calculate BIC for the model
    current_AIC <- AIC(current_model)
    
    # Print debugging information
    print(paste("Selected Features:", toString(setdiff(selected_features, feature))))
    print(paste("Excluded Feature:", feature))
    print(paste("Current AIC:", current_AIC))
    
    # Update selected features if BIC is reduced
    if (current_AIC < min_AIC) {
      min_AIC <- current_AIC
      selected_features <- setdiff(selected_features, feature)
    }
  }
}

# Display the selected features
print(selected_features)
length(selected_features)



multinom_model <- multinom(track_genre ~ ., data = df_filtered[train,c(selected_features,"track_genre")])

# Make Predictions
predictions <- predict(multinom_model, newdata = df_filtered[test,c(selected_features,"track_genre")], type = "class")

# Evaluate the Model
confusion_matrix <- table(predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Display results
print("Confusion Matrix:")
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)



#Random Forest


library(randomForest)

library(caret)

# Create a training control with cross-validation
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)
levels(df_filtered$track_genre) <- make.names(levels(df_filtered$track_genre))
# Train a Random Forest model using cross-validation
rf_model <- train(track_genre ~ ., data = df_filtered[train,],
                  method = "rf",
                  trControl = ctrl)

# Print the results
print(rf_model)



#rf_model <- randomForest(track_genre ~ ., data = df_filtered[train,])

# Make predictions on the test set
predictions <- predict(rf_model, newdata = df_filtered[test,])

# Evaluate the model
confusion_matrix <- table(predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))
print(confusion_matrix)

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)


#



#Gradient boosting algorithm


library(gbm)

#char_columns <- sapply(df_filtered[train, ], is.character)
#df_filtered[, char_columns] <- lapply(df_filtered[, char_columns], as.factor)
df_filtered$explicit <- factor(df_filtered$explicit, levels = c("False", "True"))

#train_labels <- as.numeric(y.train) - 1 
# Load the caret and gbm packages if not already loaded
# install.packages(c("caret", "gbm"))
library(caret)
library(gbm)

# Set up parameter grid for tuning (without subsample and colsample_bytree)
param_grid <- expand.grid(
  n.trees = c(50, 100, 500,1000),  # Adjust the values as needed
  interaction.depth = c(1, 2, 4, 6,8,10),
  shrinkage = c(0.01, 0.1, 0.3),
  n.minobsinnode = 10
)
param_grid <- expand.grid(
  n.trees = c(1000,1500,2000,5000),  # Adjust the values as needed
  interaction.depth = c(1, 5,10,15,20),
  shrinkage = c(0.001,0.01, 0.1),
  n.minobsinnode = 10
)



# Initialize an empty list to store results
results <- list()

# Perform grid search over hyperparameter space
for (i in 1:nrow(param_grid)) {
  # Define parameters for this iteration
  params <- list(
    n.trees = param_grid$n.trees[i],
    interaction.depth = param_grid$interaction.depth[i],
    shrinkage = param_grid$shrinkage[i],
    n.minobsinnode = param_grid$n.minobsinnode[i],
    distribution = "multinomial"  # Assuming a multiclass problem
  )
  
  # Perform cross-validation with caret and gbm
  ctrl <- trainControl(method = "cv", number = 5)
  cv_result <- train(
    x = df_filtered[train, -16],  # Exclude the response variable column
    y = as.factor(y.train),
    method = "gbm",
    trControl = ctrl,
    tuneGrid = data.frame(n.trees = params$n.trees, interaction.depth = params$interaction.depth, shrinkage = params$shrinkage, n.minobsinnode = params$n.minobsinnode)
  )
  
  # Store the results
  results[[paste0("n.trees_", params$n.trees, "_interaction_depth_", params$interaction.depth, "_shrinkage_", params$shrinkage)]] <- cv_result
}

# Get the best model
#best_index <- which.min(sapply(results, function(x) x$RMSE))
best_model <- results[[which.max(lapply(results, function(x) x$results$Accuracy))]]

# Print the best model


saveRDS(best_model, file = "best_model_gbm1.rds")

# Load the saved model back into R
loaded_model <- readRDS("best_model_gbm1.rds")

print(best_model)

# Train the best model on the full training set
final_model <- train(
  x = df_filtered[, -16],  # Exclude the response variable column
  y = as.factor(df_filtered$track_genre),
  method = "gbm",
  trControl = trainControl(method = "none"),  # No resampling for final training
  tuneGrid = data.frame(
    n.trees = best_model$bestTune$n.trees,
    interaction.depth = best_model$bestTune$interaction.depth,
    shrinkage = best_model$bestTune$shrinkage,
    n.minobsinnode = best_model$bestTune$n.minobsinnode
  )
)

# Make predictions on the test set
test_predictions <- predict(final_model, newdata = df_filtered[test,-16])

# Print or use the predictions as needed
#print(test_predictions)
confusion_matrix <- table(test_predictions, y.test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)#previous best=0.83611111
print(paste("Accuracy:", accuracy))
print(confusion_matrix)

metrics <- confusionMatrix(confusion_matrix)

# Extract precision, recall, F1-score, and specificity for each class
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]
specificity <- 1 - metrics$byClass[, "Sensitivity"]
#accuracy <- metrics$byClass[,"Accuracy"]
#str(metrics)
# Create a data frame with the metrics
metrics_df <- data.frame(
  #Class = rownames(metrics$byClass),
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score,
  Specificity = specificity
  #Accuracy = accuracy
)

# Print or use the metrics_df data frame
print(metrics_df)

importance_values <- summary(final_model)

# Print or view the variable importance values
print(importance_values)
#linear discriminant analysis
mai.old<-par()$mai
mai.old
#new vector
mai.new<-mai.old
#new space on the left
mai.new[2] <- 2.5 
mai.new
#modify graphical parameters
par(mai=mai.new)
summary(final_model, las=1) 
#las=1 horizontal names on y
summary(final_model, las=1, cBar=10) 
#cBar defines how many variables
#back to orginal window
par(mai=mai.old)


formula <- as.formula("track_genre ~ .")
data <- df_filtered[train,]  # Include the response variable in the data frame

final_model <- gbm(
  formula = formula,
  data = data,
  distribution = "multinomial",  # Specify the distribution for multiclass classification
  n.trees = best_model$bestTune$n.trees,
  interaction.depth = best_model$bestTune$interaction.depth,
  shrinkage = best_model$bestTune$shrinkage,
  n.minobsinnode = best_model$bestTune$n.minobsinnode
)

plot(final_model, i.var = 4,  main = "Partial Dependence Plot", rug = TRUE)
str(final_model)
# Manually add legend
legend("topright", legend = levels(as.factor(df_filtered$track_genre)), col = 1:9, lty = 1, cex = 0.8)


unique_classes <- levels(df_filtered$track_genre)

# Plot partial dependence for each class
for (i in seq_along(unique_classes)) {
  class_name <- unique_classes[i]
  
  # Calculate partial dependence
  pd <- partial(final_model, pred.var = 1, grid.resolution = 100,n.trees=2000, plot = FALSE, class = i)
  
  # Create custom plot
  pdp::plotPartial(pd, plot.title = "Partial Dependence Plot")

}


library(pdp)

# Create partial dependence plot
pdp_results <- partial(final_model, pred.var = 7, grid.resolution = 100,n.trees=2000)

# Plot partial dependence with legend
pdp::plotPartial(pdp_results, plot.title = "Partial Dependence Plot", col.lines = 1:9, legend.names = levels(as.factor(df_filtered$track_genre)))







#svm



#sparse svm






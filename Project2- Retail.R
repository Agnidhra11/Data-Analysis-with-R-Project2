df_store_train=read.csv("store_train.csv", stringsAsFactors = F)
df_store_test=read.csv("store_test.csv", stringsAsFactors = F)

getwd()
setwd("C:/Users/Agnidhra Banerjee/Desktop")

View(df_store_train)

df_store_train[df_store_train$CouSub==99999,9]=NA

round(colMeans(is.na(df_store_train)),2)

tapply(df_housing_train$Price, df_housing_train[,7],sum)

tapply(df_store_train[,c(2,3,4,5,6)],df_store_train[df_store_train=="Grocery Store",],var)

sort(table(df_store_train$Areaname), decreasing = FALSE)

x=df_store_train[df_store_train$Areaname=="Kennebec County, ME",]
View(x)
y=x[x$store_Type=="Supermarket Type1",]
View(y)
lapply(y[,c(2,3,4,5,6)],sum)

unique(df_store_train$Areaname)
table(df_store_train$store_Type)

x=df_store_train[df_store_train$store_Type=="Supermarket Type3",]
View(x)
lapply(x[,c(2,3,4,5,6)],var)
table(x$store)
182/432

sort(table(df_store_train$state_alpha), decreasing = T)

shapiro.test(df_store_train$sales1)


sort(table(df_store_train$country),decreasing = T)
sort(table(df_store_train$State),decreasing = T)
sort(table(df_store_train$state_alpha),decreasing = T)
sort(table(df_store_train$countyname),decreasing = T)[1:10]
sort(table(df_store_train$countytownname),decreasing = T)[1:10]
sort(table(df_store_train$Areaname),decreasing = T)[1:10]

glimpse(df_store_train)
df_store_train$country=as.character(df_store_train$country)
df_store_train$store=as.factor(df_store_train$store)

dp_pipe=recipe(store ~ .,data=df_store_train) %>% 
  update_role(Id,State,storecode,countyname, new_role = "drop_vars") %>% 
  update_role(country,state_alpha,store_Type,Areaname,countytownname,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.005,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data = NULL)

test=bake(dp_pipe,new_data=df_store_test)

lapply(train,function(x) sum(is.na(x)))

set.seed(2)
s=sample(1:nrow(train),0.8*nrow(train))
t1=train[s,]
t2=train[-s,]

## dtree

tree_model=decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")


folds = vfold_cv(train, v = 20)


tree_grid = grid_regular(cost_complexity(), tree_depth(),
                         min_n(), levels = 4)

# doParallel::registerDoParallel()
my_res=tune_grid(
  tree_model,
  store~.,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
  
)

autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_tree_fit=tree_model %>% 
  finalize_model(select_best(my_res)) %>% 
  fit(store~.,data=train)

# feature importance 



final_tree_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# plot the tree

rpart.plot(final_tree_fit$fit)

# predictions

train_pred=predict(final_tree_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_tree_fit,new_data = test,type="prob") %>% select(.pred_1)

write.csv(test_pred,"Agnidhra_Banerjee_P2_part2.csv",row.names = F)

### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$store

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit,legend=F)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)


## Random Forest

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(t1, v = 10)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),
                       min_n(c(2,10)),levels = 3)
View(t1)
glimpse(t1)

my_res=tune_grid(
  rf_model,
  store~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(store~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)

write.csv(test_pred,"Agnidhra_Banerjee_P2_part2.csv",row.names = F)
### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$store

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)


write.csv(test_hard_class,"Agnidhra_Banerjee_P2_part2.csv",row.names = F)


for_vif=lm(store~.,data=t1)

sort(vif(for_vif),decreasing = T)[1:3]

log_fit=glm(store~.-store_Type_X__other__ -sales0 -sales2 
            -countyname_X__other__ -sales3 -state_alpha_ME -sales1,data=t1,
            family = "binomial")
summary(log_fit)

log_fit=stats::step(log_fit)

formula(log_fit)

log_fit=glm(store ~ sales4 + population + country_X11 +  country_X123 + 
              country_X13 +   country_X5 + country_X51 + 
               country_X61 +  countyname_Berkshire.County + 
               countyname_Carroll.County + countyname_Cheshire.County + 
               countyname_Grafton.County + 
               
              countyname_Litchfield.County + countyname_Merrimack.County + 
              countyname_Orleans.County + countyname_Penobscot.County + 
               countyname_Rockingham.County + 
               countyname_York.County +  
               state_alpha_CT + state_alpha_GA +  
               state_alpha_IN +  state_alpha_KY + 
              state_alpha_LA + state_alpha_MA +  state_alpha_MO + 
                state_alpha_OK + state_alpha_PR + 
              state_alpha_RI +  state_alpha_TN + state_alpha_TX + 
               state_alpha_VT + state_alpha_WI + state_alpha_WV,
            data = t1, family = "binomial")

summary(log_fit)

val.score=predict(log_fit,newdata = t2,type='response')
real=t2$store

pROC::auc(pROC::roc(real,train.score))

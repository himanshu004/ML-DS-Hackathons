# using pipelining for multi model tuning
pipe = Pipeline([('classifier' , LogisticRegression())])

# Param grid for multi - model tuning (using pipeline)
param_grid = [
    {
    'classifier' : [model1],
    'classifier__penalty': ['l1','l2','elasticnet'],
    'classifier__C': np.linspace(1,15,30),
    'classifier__solver': ['liblinear'],
    'classifier__random_state': [1,2,3,4,5,6,7]
     },
    {
    'classifier' : [model5],
    'classifier__activation': ['tanh', 'relu','logistic'],
    'classifier__solver': ['sgd', 'adam'],
    'classifier__alpha': [0.0001, 0.0002, 0.0004,0.001,0.002,0.01],
    'classifier__learning_rate': ['constant','adaptive'],
    'classifier__hidden_layer_sizes': [(30,40,30), (40,40,40,40,40)]
    }

]

# Grid Search Cross Validation
gscv_pipe = GridSearchCV(pipe, param_grid = param_grid, cv = 10, verbose=False, n_jobs=-1)

# Randomized Search Cross Validation
rscv_pipe = RandomizedSearchCV(pipe, param_distributions = param_grid, n_iter = 100, cv = 10, verbose=False, random_state=42, n_jobs = -1)

print('Best estimator using grid-searchCV: ',gscv_pipe.best_estimator_)
gscv_pipe.fit(x_train,y_train)
y_predict3 = gscv_pipe.best_estimator_.predict(x_test)

print('Best estimator using random grid-searchCV: ',rscv_pipe.best_estimator_)
rscv_pipe.fit(x_train,y_train)
y_predict4 = rscv_pipe.best_estimator_.predict(x_test)



gscv_pipe_submit = pd.DataFrame(y_predict3,columns = ['Survived'],index = x_test.index + 892)
rscv_pipe_submit = pd.DataFrame(y_predict4,columns = ['Survived'],index = x_test.index + 892)

gscv_pipe_submit.index.name = 'PassengerId'
gscv_pipe_submit

rscv_pipe_submit.index.name = 'PassengerId'
rscv_pipe_submit

gscv_pipe_submit.to_csv('gscv_pipe_submission.csv')
rscv_pipe_submit.to_csv('rscv_pipe_submission.csv')
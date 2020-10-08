pipe = Pipeline([('classifier' , LogisticRegression())])

# # Param grid for multi - model tuning (using pipeline)
param_grid = [
    {
    'classifier' : [model1],
    'classifier__penalty': ['l1','l2','elasticnet'],
    'classifier__C': np.linspace(1,15,30),
    'classifier__solver': ['liblinear'],
    'classifier__random_state': [1,2,3,4,5,6,7]
     },
    # {'classifier' : [RandomForestClassifier()],
    #  'classifier__criterion': ['gini', 'entropy'],
    #             'classifier__bootstrap': [True, False],
    #             #  'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #             #  'classifier__max_features': ['auto', 'sqrt'],
    #             #  'classifier__min_samples_leaf': [1, 2, 4],
    #             #  'classifier__min_samples_split': [2, 5, 10],
    #              'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    # }
    {
    'classifier' : [model5]
    'classifier__hidden_layer_sizes': [(30,40,30), (40,40,40,40,40)],
    'classifier__activation': ['tanh', 'relu','logistic'],
    'classifier__solver': ['sgd', 'adam'],
    'classifier__alpha': [0.0001, 0.0002, 0.0004,0.001,0.002,0.01],
    'classifier__learning_rate': ['constant','adaptive'],
    }

]


import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
import numpy as np  

# import pre-processing functions
from preprosess_pipeline.pre_processing import load_and_process_data, build_preprocessing_pipeline


# configure mlflow  track experiments locally
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart_Disease_project_experiments")

def log_matrices(y_test, y_pred, y_prob=None, model_name="model"):
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    if y_prob is not None:
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
    print(f"logged {model_name} metrics to mlflow")
    

def select_best_cv(model_dict, X_train, y_train):
    print('Selecting the best model based on cross-validation scores... ')
    
    scoring_matrices = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_result_summary = {}
    
    for model_name, model in model_dict.items():
        # 5-fold cross-validation
        cv_scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring_matrices)
        
        mean_recall = np.mean(cv_scores['test_recall'])
        mean_precision = np.mean(cv_scores['test_precision'])
        mean_f1 = np.mean(cv_scores['test_f1'])
        mean_accuracy = np.mean(cv_scores['test_accuracy'])
        mean_roc_auc = np.mean(cv_scores['test_roc_auc']) 
        
        cv_result_summary[model_name] = {
            'model': model,
            'mean_recall': mean_recall,
            'mean_precision': mean_precision,
            'mean_f1': mean_f1,
            'mean_accuracy': mean_accuracy,
            'mean_roc_auc': mean_roc_auc
        }   
        
        print(f"model: {model_name} | mean_recall: {mean_recall:.4f} | mean_precision: {mean_precision:.4f} | mean_f1: {mean_f1:.4f} | mean_accuracy: {mean_accuracy:.4f} | mean_roc_auc: {mean_roc_auc:.4f}")
        
        # sort models based on recall, then roc_auc, then f1, then precision, then accuracy
        ranked_models = sorted(
            cv_result_summary.items(), 
            key=lambda x: (
                x[1]['mean_recall'], 
                x[1]['mean_roc_auc'], 
                x[1]['mean_f1'], 
                x[1]['mean_precision'], 
                x[1]['mean_accuracy']
            ), 
            reverse=True
        ) 
        # unpack the best model name and its info
        winner_model_name, winner_model_info = ranked_models[0] 
    print(f"\nBest model based on CV scores: {winner_model_name} with mean_recall: {winner_model_info['mean_recall']:.4f}, mean_roc_auc: {winner_model_info['mean_roc_auc']:.4f}, mean_f1: {winner_model_info['mean_f1']:.4f}, mean_precision: {winner_model_info['mean_precision']:.4f}, mean_accuracy: {winner_model_info['mean_accuracy']:.4f}")
    
    # Serialize the best model using joblib
    joblib.dump(winner_model_info['model'], f"best_{winner_model_name}.pkl")
    print(f"best {winner_model_name} model serialized to disk")
    
  
  
def train_and_log_models():
    # call functions imported from pre_processing 
    X_train, X_test, y_train, y_test = load_and_process_data()
    preprocessor = build_preprocessing_pipeline()
    
    # Experiment 1: Logistic Regression
    with mlflow.start_run(run_name="Tuned_Logistic_Regression"):
        LR_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        param_grid_LR = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        }

        grid_search_LR = GridSearchCV(LR_pipeline, param_grid_LR, cv=5, n_jobs=-1)
        grid_search_LR.fit(X_train, y_train)
        
        # best_LR_pipeline = grid_search_LR.best_estimator_
        best_LR_model = grid_search_LR.best_estimator_
        y_predict_LR = best_LR_model.predict(X_test)
        y_prob_LR = best_LR_model.predict_proba(X_test)[:, 1] # 1 here is the probability of the positive class (heart disease present)
        
        
        # Log Logistic Regression metrics
        mlflow.log_params(grid_search_LR.best_params_)
        log_matrices(y_test, y_predict_LR, y_prob_LR, model_name="Logistic_Regression")
        # Log model
        mlflow.sklearn.log_model(best_LR_model, "Tuned_logistic_regression_model")
        print("logged logistic regression model and metrics to mlflow")
        

    # Experiment 2: Random Forest
    with mlflow.start_run(run_name="Tuned_Random_Forest"):
        RF_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        param_grid_RF = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
            }
        
        grid_search_RF = GridSearchCV(RF_pipeline, param_grid_RF, cv=5, n_jobs=-1)
        grid_search_RF.fit(X_train, y_train)
        best_RF_model = grid_search_RF.best_estimator_
        y_pred_RF = best_RF_model.predict(X_test)

        # Log Random Forest metrics
        mlflow.log_params(grid_search_RF.best_params_)
        log_matrices(y_test, y_pred_RF, model_name="Tuned_Random_Forest")
        # Log model
        mlflow.sklearn.log_model(best_RF_model, "Tuned_random_forest_model")
        print("logged random forest model and metrics to mlflow")
        
        # ----------- model comparison -----------
        print("\nModel Comparison:")
        model_dict = {
            "Tuned_Logistic_Regression": best_LR_model,
            "Tuned_Random_Forest": best_RF_model
        }
        
        # select the best model based on cross-validation scores
        select_best_cv(model_dict, X_train, y_train)
        
        
if __name__ == "__main__":
    train_and_log_models()
        
        
        
    
    


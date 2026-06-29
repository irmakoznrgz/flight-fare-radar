import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import time

from data_preprocessing import *

def train_base_model(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.1,
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    duration = end_time - start_time
    return model, duration

def train_optimize_model(X_train, y_train):
    base_model = LGBMRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    }
    
    #The reason 'neg' is used before 'neg_mean_absolute_error' is that the lower the mean absolute error (MAE), the better the model.
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,         
        n_jobs=-1,   
        verbose=1  
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    duration = end_time - start_time
    print("="*60)
    print(f"Best parameters found: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, duration

def evaluate_duel(base_model, opt_model, base_time, opt_time, X_test, y_test):

    y_pred_b = base_model.predict(X_test)
    mae_base = mean_absolute_error(y_test, y_pred_b)
    r2_base = r2_score(y_test, y_pred_b)
    y_pred_opt = opt_model.predict(X_test)
    mae_opt = mean_absolute_error(y_test, y_pred_opt)
    r2_opt = r2_score(y_test, y_pred_opt)
    
    print("="*60)
    print(f"'Base Model R-Squared (R2) Score: {r2_base:.4f}")
    print(f"Opt Model R-Squared (R2) Score: {r2_opt:.4f}")
    print("-"*60)
    print(f"Base Model Mean Absolute Error (MAE): {mae_base:.2f}")
    print(f"Opt Model Mean Absolute Error (MAE):{mae_opt:.2f}")
    print("-"*60)
    print(f"Base Model Training Time: {base_time:.2f}")
    print(f"Opt Model Training Time: {opt_time:.2f}")
    print("="*60 + "\n")
    
    if mae_opt < mae_base:
        print("Winner: GridSearchCV Model")
        return opt_model
    else:
        print("Winner: Base Model")
        return base_model

def save_best_model(model, file_path="models/flight_price_radar.pkl"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model successfully saved to system: {file_path}")

if __name__ == "__main__":
    raw_data_path = "data/raw/flights.csv"
    
    df_clean = load_and_clean_data(raw_data_path)
    df_processed = perform_feature_engineering(df_clean)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    base_model, base_time = train_base_model(X_train, y_train)
    
    opt_model, opt_time = train_optimize_model(X_train, y_train)
    
    best_champ = evaluate_duel(base_model, opt_model, base_time, opt_time, X_test, y_test)
    
    save_best_model(best_champ)
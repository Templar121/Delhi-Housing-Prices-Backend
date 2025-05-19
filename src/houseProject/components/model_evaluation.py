import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score
from urllib.parse import urlparse
import numpy as np
import joblib
from pathlib import Path
from houseProject.entity.config_entity import ModelEvaluationConfig
from houseProject.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def evaluate_model(self, actual, pred):
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        r2_square = r2_score(actual, pred)
        med_ae = median_absolute_error(actual, pred)
        evs = explained_variance_score(actual, pred)
        
        return mae, mse, rmse, r2_square, med_ae, evs

    


    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        (mae, mse, rmse, r2, med_ae, evs) = self.evaluate_model(test_y, predicted_qualities)
        
        # Saving metrics as local
        scores = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Score': r2, 'Median AE': med_ae, 'Explained Variance Score': evs}
        save_json(path=Path(self.config.metric_file_name), data=scores)
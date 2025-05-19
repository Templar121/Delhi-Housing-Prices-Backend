import pandas as pd
import os
from houseProject import logger
from catboost import CatBoostRegressor
import joblib
from houseProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Split features and target
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Initialize RandomForestRegressor
        model = CatBoostRegressor(
            verbose=self.config.verbose,
            random_seed=self.config.random_seed,
        )

        # Train the model
        model.fit(train_x, train_y)

        # Save the model
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
import numpy as np
import pandas as pd
from config import config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        self.numeric_features = None
        self.categorical_features = None
        self.preprocessor = None

    def _build_preprocessor(self, df: pd.DataFrame):
        numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if config.TARGET in numeric_features:
            numeric_features.remove(config.TARGET)
        if config.TARGET in categorical_features:
            categorical_features.remove(config.TARGET)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, train_df: pd.DataFrame):
        X_train = train_df.drop(config.TARGET, axis=1, errors='ignore')
        self._build_preprocessor(X_train)
        self.preprocessor.fit(X_train)

    def transform(self, df: pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized, call fit() first.")
        
        X_processed = self.preprocessor.transform(df)
        return X_processed

    def fit_transform(self, train_df: pd.DataFrame):
        X_train = train_df.drop(config.TARGET, axis=1, errors='ignore')
        y_train = train_df[config.TARGET] if config.TARGET in train_df else None

        self.fit(train_df)
        X_train_processed = self.transform(X_train)

        return X_train_processed, y_train

    def get_feature_names(self):
        if self.preprocessor is None:
            return None
        
        feature_names = []
        for name, trans, cols in self.preprocessor.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                feature_names.extend(trans.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)

        return feature_names

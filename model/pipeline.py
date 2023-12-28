from feature_engine.encoding import CountFrequencyEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from model.config.core import config

# Creating the pipeline

scaler = MinMaxScaler()
encoder = CountFrequencyEncoder()
params = {"n_estimators": 180, "max_depth": 14}
classifier = RandomForestClassifier()

# Preprocessing steps for the numerical variables
num_preproc = Pipeline(steps=[("scaler", scaler)])

# Preprocessing steps for the categorical variables
cat_preproc = Pipeline(steps=[("encoder", encoder)])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat_preprocessor", cat_preproc, config.model_config.categorical_vars),
        ("numerical_preprocessor", num_preproc, config.model_config.numerical_vars),
    ]
)

pipe = Pipeline(steps=[("preprocessing", preprocessor), ("model", classifier)])

from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import CategoricalImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from classification_model.config.core import config
from classification_model.processing.features import AuxiliaryFunctions

af = AuxiliaryFunctions()


classifier_pipeline = Pipeline(
    [
        # Change format for days_birth and days_employed
        (
            "time_func",
            SklearnTransformerWrapper(
                transformer=(FunctionTransformer(af.get_years)),
                variables=config.model_config.numerical_features_to_years,
            ),
        ),
        # missing data imputation
        (
            "imputer_cat",
            CategoricalImputer(
                variables=config.model_config.categorical_features_with_na_missing_and_rare_label
            ),
        ),
        # Rare labels
        (
            "encoder_rare_label",
            RareLabelEncoder(
                tol=config.model_config.rare_label_tol,
                n_categories=config.model_config.rare_label_n_categories,
                variables=config.model_config.categorical_features_with_na_missing_and_rare_label,
            ),
        ),
        # categorical encoding
        (
            "categorical_encoder",
            OrdinalEncoder(
                encoding_method=config.model_config.categorical_encoder_method,
                variables=config.model_config.categorical_features_encoding,
            ),
        ),
        # Random Forest Classifier
        (
            "rfc",
            RandomForestClassifier(
                n_estimators=config.model_config.rfc_n_estimators,
                criterion=config.model_config.rfc_criterion,
                max_depth=config.model_config.rfc_max_depth,
                min_samples_split=config.model_config.rfc_min_samples_split,
                min_samples_leaf=config.model_config.rfc_min_samples_leaf,
                random_state=config.model_config.random_state,
                class_weight=config.model_config.rfc_class_weight,
            ),
        ),
    ]
)

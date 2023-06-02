from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import FunctionTransformer

from classification_model.config.core import config
from classification_model.processing.features import AuxiliaryFunctions


def test_get_year(sample_input_data):
    # Given
    sample_input_data.rename(
        columns=config.model_config.variables_to_rename, inplace=True
    )

    transformer = SklearnTransformerWrapper(
        transformer=(FunctionTransformer(AuxiliaryFunctions().get_years)),
        variables=config.model_config.numerical_features_to_years,
    )

    assert (
        sample_input_data[config.model_config.numerical_features_to_years].iat[0, 0]
        == -20145
    )

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject[config.model_config.numerical_features_to_years].iat[0, 0] == 55

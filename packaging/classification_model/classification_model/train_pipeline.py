from config.core import config
from pipeline import classifier_pipeline
from processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # fit model
    classifier_pipeline.fit(X_train, y_train)

    # evaluate model
    y_train_pred = classifier_pipeline.predict_proba(X_train)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    y_test_pred = classifier_pipeline.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, y_test_pred)
    print(f"ROC AUC Training = {roc_auc_train}")
    print(f"ROC AUC Test = {roc_auc_test}")

    # persist trained model
    save_pipeline(pipeline_to_persist=classifier_pipeline)


if __name__ == "__main__":
    run_training()

from sklearn.model_selection import train_test_split

from model.config.core import config
from model.pipeline import pipe
from model.preprocessing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """
    Train the model.

    Training data can be found here:
    https://www.openml.org/data/get_csv/16826755/phpMYEkMl
    """

    # read training data
    data = load_dataset(
        client_file_name=config.app_config.client_data_file,
        price_file_name=config.app_config.price_data_file,
    )

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
    pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=pipe)


if __name__ == "__main__":
    run_training()

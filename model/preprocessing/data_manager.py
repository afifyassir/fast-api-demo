import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

logger = logging.getLogger(__name__)


def load_dataset(*, client_file_name: str, price_file_name: str) -> pd.DataFrame:
    # Loading the datasets.
    dataframe_client = pd.read_csv(Path(f"{DATASET_DIR}/{client_file_name}"))
    dataframe_price = pd.read_csv(Path(f"{DATASET_DIR}/{price_file_name}"))

    # Converting to datetime.
    dataframe_client = datetime_conversion_client(df=dataframe_client)
    dataframe_price = datetime_conversion_price(df=dataframe_price)

    # Transforming the price data.
    dataframe_price = price_data_trans(df=dataframe_price)

    # Merging the client and price datasets.
    dataframe = merging_datasets(df=dataframe_client, df_1=dataframe_price)

    # Getting time and consumption features.
    dataframe = time_features(df=dataframe)
    dataframe = consum_features(df=dataframe)

    dataframe = dataframe.drop("Unnamed: 0", axis=1)
    dataframe = dataframe.dropna()

    # Creating the new categorical features.
    dataframe["price_change_energy"] = dataframe[
        "offpeak_diff_dec_january_energy"
    ].apply(price_change)
    dataframe["price_change_power"] = dataframe["offpeak_diff_dec_january_power"].apply(
        price_change
    )

    # Dropping the first feature.
    dataframe = dataframe.drop(
        ["offpeak_diff_dec_january_energy", "offpeak_diff_dec_january_power"], axis=1
    )

    dataframe = dataframe.dropna()

    return dataframe


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    return joblib.load(filename=file_path)


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def datetime_conversion_client(df: pd.DataFrame) -> pd.DataFrame:

    df["date_activ"] = pd.to_datetime(df["date_activ"], format="%Y-%m-%d")
    df["date_end"] = pd.to_datetime(df["date_end"], format="%Y-%m-%d")
    df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format="%Y-%m-%d")
    df["date_renewal"] = pd.to_datetime(df["date_renewal"], format="%Y-%m-%d")

    return df


def datetime_conversion_price(df: pd.DataFrame) -> pd.DataFrame:

    df["price_date"] = pd.to_datetime(df["price_date"], format="%Y-%m-%d")

    return df


def price_data_trans(df: pd.DataFrame) -> pd.DataFrame:

    # Group off-peak prices by companies and month
    monthly_price_by_id = (
        df.groupby(["id", "price_date"])
        .agg({"price_off_peak_var": "mean", "price_off_peak_fix": "mean"})
        .reset_index()
    )

    # Get january and december prices
    jan_prices = monthly_price_by_id.groupby("id").first().reset_index()
    dec_prices = monthly_price_by_id.groupby("id").last().reset_index()

    # Calculate the difference
    diff = pd.merge(
        dec_prices.rename(
            columns={"price_off_peak_var": "dec_1", "price_off_peak_fix": "dec_2"}
        ),
        jan_prices.drop(columns="price_date"),
        on="id",
    )
    diff["offpeak_diff_dec_january_energy"] = diff["dec_1"] - diff["price_off_peak_var"]
    diff["offpeak_diff_dec_january_power"] = diff["dec_2"] - diff["price_off_peak_fix"]
    diff = diff[
        ["id", "offpeak_diff_dec_january_energy", "offpeak_diff_dec_january_power"]
    ]

    return diff


def merging_datasets(df: pd.DataFrame, df_1: pd.DataFrame) -> pd.DataFrame:

    new_df = df.merge(df_1, how="left", left_on="id", right_on="id")

    return new_df


# A function to create the new feature "price_change"
def price_change(x: float) -> str:
    if x > 0:
        return "increase"
    elif x < 0:
        return "decrease"
    else:
        return "stable"


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Getting the activation year
    df["activ_year"] = df["date_activ"].dt.year

    # Getting the activation month
    df["activ_month"] = df["date_activ"].dt.month

    # Getting the ending year
    df["end_year"] = df["date_end"].dt.year

    # Getting the ending month
    df["end_month"] = df["date_end"].dt.month

    # Getting the year of the last modification of the product
    df["modif_prod_year"] = df["date_modif_prod"].dt.year

    # Getting the month of the last modification of the product
    df["modif_prod_month"] = df["date_modif_prod"].dt.month

    # Getting the renewal year
    df["renewal_year"] = df["date_renewal"].dt.year

    # Getting the renewal month
    df["renewal_month"] = df["date_renewal"].dt.month

    # Difference between activation and renewal in days
    df["diff_act_renew"] = (df["date_renewal"] - df["date_activ"]).dt.days

    # Difference between activation and end in days
    df["diff_act_end"] = (df["date_end"] - df["date_activ"]).dt.days

    # Difference between activation and production modification.
    df["diff_act_modif"] = (df["date_modif_prod"] - df["date_activ"]).dt.days

    # Difference between end and production modification.
    df["diff_end_modif"] = (df["date_end"] - df["date_modif_prod"]).dt.days

    return df


def consum_features(df: pd.DataFrame) -> pd.DataFrame:

    # Getting the average consumption per month for the past 12 months.
    df["avrg_month_cons"] = df["cons_12m"] / 12

    # Getting the average consumption per month of gaz for the past 12 months.
    df["avrg_month_cons_gaz"] = df["cons_gas_12m"] / 12

    # Getting the average forcasted consumption per month for the next 12 months.
    df["forcast_avrg_month_cons"] = df["forecast_cons_12m"] / 12

    # Getting the ratio of the last month consumption to the last 12m consumption
    df["ratio_last_month_last12m_cons"] = df["cons_last_month"] / df["cons_12m"]

    # Getting the ratio of the last month consumption to the the average consumption per month for the past 12 months.
    df["ratio_last_month_avg_cons"] = df["cons_last_month"] / df["avrg_month_cons"]

    return df

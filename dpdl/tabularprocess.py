import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Tuple


def preprocess_adult_census_income(df, normalize_features=True):
    """Preprocess the Adult Census Income dataset."""
    df = df.dropna()
    df = df.drop("fnlwgt", axis=1)
    df["race"] = df["race"].apply(
        lambda x: "white" if x.strip() == "White" else "non-white"
    )

    # Normalize numerical features
    num_features = [
        "age",
        "educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    if normalize_features:
        scaler = StandardScaler()
        df[num_features] = scaler.fit_transform(df[num_features])

    # Encode categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
        "gender",
        "income",
    ]
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_features = pd.DataFrame(
        encoder.fit_transform(df[cat_features]),
        columns=encoder.get_feature_names_out(cat_features),
    )
    df = pd.concat([df.drop(cat_features, axis=1), encoded_features], axis=1)
    return df


def preprocess_dutch_dataset(df):
    """Preprocess the Dutch dataset."""
    df["occupation"] = df["occupation"].apply(lambda x: 1 if x == "2_1" else 0)
    df["gender"] = df["gender"].apply(lambda x: 1 if x == 1 else 0)
    df = df.dropna()

    cat_features = [
        "age_group",
        "household_size",
        "household_position",
        "prev_residence_place",
        "citizenship",
        "country_birth",
        "edu_level",
        "economic_status",
        "cur_eco_activity",
        "Marital_status",
    ]

    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_features = pd.DataFrame(
        encoder.fit_transform(df[cat_features]),
        columns=encoder.get_feature_names_out(cat_features),
    )
    df = pd.concat([df.drop(cat_features, axis=1), encoded_features], axis=1)
    return df


def balance_classes(df, base_column, target_amount=14000):
    df_base_positive = df[df[base_column] == 1]
    df_base_negative = df[df[base_column] == 0]

    number_of_simples = min(len(df_base_positive), len(df_base_negative), target_amount)
    df_base_positive_sample = df_base_positive.sample(
        min(len(df_base_positive), number_of_simples)
    )
    df_base_negative_sample = df_base_negative.sample(
        min(len(df_base_negative), number_of_simples)
    )

    df_balanced = pd.concat([df_base_positive_sample, df_base_negative_sample], axis=0)
    df_test = df.drop(df_balanced.index)

    return df_balanced.sample(frac=1), df_test

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the house price prediction dataset.
    """

    df = df.copy()

    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    df["area_per_bedroom"] = df["area"] / df["bedrooms"].replace(0, 1)

    df["area_per_bathroom"] = df["area"] / df["bathrooms"].replace(0, 1)

    df["area_per_room"] = df["area"] / df["total_rooms"].replace(0, 1)

    df["parking_density"] = df["parking"] / df["total_rooms"].replace(0, 1)

    df["has_parking"] = (df["parking"] > 0).astype(int)

    return df
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create extra useful features from existing housing data.

    These features help the machine learning model understand
    house size, room density, and parking availability better.
    """

    df = df.copy()

    # Total rooms = bedrooms + bathrooms
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    # Area per bedroom
    # replace(0, 1) avoids division by zero
    df["area_per_bedroom"] = df["area"] / df["bedrooms"].replace(0, 1)

    # Area per bathroom
    df["area_per_bathroom"] = df["area"] / df["bathrooms"].replace(0, 1)

    # Parking availability
    # 1 means parking available, 0 means no parking
    df["has_parking"] = df["parking"].apply(lambda x: 1 if x > 0 else 0)

    return df
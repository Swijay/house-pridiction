import os
import pandas as pd


# Input and output paths
RAW_DATA_PATH = "data/houses.csv"
PROCESSED_DATA_PATH = "data/houses.parquet"


# Expected column data types
DTYPE_SCHEMA = {
    "price": "int64",
    "area": "int64",
    "bedrooms": "int64",
    "bathrooms": "int64",
    "stories": "int64",
    "mainroad": "string",
    "guestroom": "string",
    "basement": "string",
    "hotwaterheating": "string",
    "airconditioning": "string",
    "parking": "int64",
    "prefarea": "string",
    "furnishingstatus": "string",
}


# Required columns in the dataset
REQUIRED_COLUMNS = list(DTYPE_SCHEMA.keys())


def validate_columns(df: pd.DataFrame) -> None:
    """
    Check whether all required columns are present in the dataset.
    """

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean categorical text columns by removing extra spaces
    and converting values to lowercase.
    """

    categorical_columns = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus",
    ]

    for col in categorical_columns:
        df[col] = df[col].astype("string").str.strip().str.lower()

    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce correct data types for all columns.
    """

    for column, dtype in DTYPE_SCHEMA.items():
        df[column] = df[column].astype(dtype)

    return df


def ingest_house_data() -> None:
    """
    Main ingestion function:
    1. Reads raw CSV file
    2. Validates columns
    3. Cleans categorical columns
    4. Enforces data types
    5. Removes duplicates
    6. Writes data as Parquet
    """

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")

    print("Reading raw house data...")

    df = pd.read_csv(RAW_DATA_PATH)

    print("Raw data loaded successfully.")
    print(f"Raw shape: {df.shape}")

    validate_columns(df)

    df = df[REQUIRED_COLUMNS]

    df = clean_categorical_columns(df)

    df = enforce_dtypes(df)

    df = df.drop_duplicates()

    os.makedirs("data", exist_ok=True)

    df.to_parquet(PROCESSED_DATA_PATH, index=False)

    print("Data ingestion completed successfully.")
    print(f"Final shape: {df.shape}")
    print(f"Parquet file saved at: {PROCESSED_DATA_PATH}")

    print("\nFinal data types:")
    print(df.dtypes)


if __name__ == "__main__":
    ingest_house_data()
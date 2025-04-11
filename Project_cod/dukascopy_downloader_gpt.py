import os
import pandas as pd
import click
from datetime import datetime
from duka.core.utils import TimeFrame


# --- Downloading using duka ---
def download_asset(symbol, start_date, end_date, timeframe, folder):
    """
    Download candle data for a single asset using duka.
    Always uses candle mode (e.g. M1, H1, or D1).
    """
    # Map the timeframe string to the duka TimeFrame enum.
    timeframes = {"M1": TimeFrame.M1, "H1": TimeFrame.H1, "D1": TimeFrame.D1}
    try:
        # Import the callable "app" from duka.app (the correct entry-point)
        from duka.app import app as duka_app

        # Call the duka app with the candle parameter (which is the fifth parameter)
        duka_app(
            symbols=[symbol],
            start=start_date,
            end=end_date,
            threads=10,
            timeframe=timeframes[timeframe],  # this tells duka to download candle data
            folder=folder,
            header=True,
        )
    except Exception as e:
        print(f"Error downloading {symbol} data: {e}")


# --- Processing and merging CSV files ---
def process_file(file_path, symbol):
    """
    Process a downloaded CSV file: rename columns and convert the
    'Timestamp' column (in ms) to a UTC datetime index.
    If the file is empty or does not contain 'Timestamp', return an empty DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

    if "Timestamp" not in df.columns:
        print(f"File {file_path} does not contain 'Timestamp'; skipping.")
        return pd.DataFrame()

    df.rename(
        columns={
            "Open": f"{symbol}_open",
            "High": f"{symbol}_high",
            "Low": f"{symbol}_low",
            "Close": f"{symbol}_close",
            "Volume": f"{symbol}_volume",
        },
        inplace=True,
    )
    # Convert 'Timestamp' (milliseconds) to UTC datetime index
    df["time"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df.set_index("time", inplace=True)
    df.drop(columns=["Timestamp"], inplace=True)
    return df[
        [
            f"{symbol}_open",
            f"{symbol}_high",
            f"{symbol}_low",
            f"{symbol}_close",
            f"{symbol}_volume",
        ]
    ]


def merge_datasets(folder, symbols, output_file):
    """
    For each symbol, find the latest downloaded CSV file (based on the encoded end date in the filename),
    process it, and merge all symbols’ data on the datetime index.
    """
    merged_df = pd.DataFrame()
    for symbol in symbols:
        symbol_lower = symbol.lower()
        files = [
            f
            for f in os.listdir(folder)
            if f.lower().startswith(symbol_lower) and f.endswith(".csv")
        ]
        if not files:
            print(f"No files found for {symbol} in {folder}.")
            continue

        def parse_end_date(filename):
            try:
                # Expected filename format:
                #   "{symbol}-{timeframe}_{start_year}_{start_month}_{start_day}-{end_year}_{end_month}_{end_day}.csv"
                parts = filename.split("-")
                if len(parts) >= 2:
                    end_part = parts[1].replace(".csv", "")
                    return datetime.strptime(end_part, "%Y_%m_%d")
            except Exception as e:
                print(f"Error parsing date from {filename}: {e}")
            return datetime.min

        latest_file = max(files, key=parse_end_date)
        df = process_file(os.path.join(folder, latest_file), symbol_lower)
        if df.empty:
            print(f"Processed file {latest_file} for {symbol} is empty; skipping.")
            continue
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(
                merged_df, df, how="outer", left_index=True, right_index=True
            )
    merged_df.ffill().sort_index(inplace=True)
    merged_df.to_csv(output_file)
    return merged_df


def extract_close_prices(folder, symbols, output_file):
    """
    For each symbol, extract only the close price column from its latest CSV file,
    merge these into a single DataFrame, and save to CSV.
    """
    close_df = pd.DataFrame()
    for symbol in symbols:
        symbol_lower = symbol.lower()
        files = [
            f
            for f in os.listdir(folder)
            if f.lower().startswith(symbol_lower) and f.endswith(".csv")
        ]
        if not files:
            print(f"No files found for {symbol} in {folder}.")
            continue

        def parse_end_date(filename):
            try:
                parts = filename.split("-")
                if len(parts) >= 2:
                    end_part = parts[1].replace(".csv", "")
                    return datetime.strptime(end_part, "%Y_%m_%d")
            except Exception as e:
                print(f"Error parsing date from {filename}: {e}")
            return datetime.min

        latest_file = max(files, key=parse_end_date)
        try:
            df = pd.read_csv(os.path.join(folder, latest_file))
        except Exception as e:
            print(f"Error reading file {latest_file}: {e}")
            continue
        if "Timestamp" not in df.columns:
            print(f"File {latest_file} does not contain 'Timestamp'; skipping.")
            continue
        df.rename(columns={"Close": f"{symbol_lower}_close"}, inplace=True)
        df["time"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        df.drop(columns=["Timestamp"], inplace=True)
        df_close = df[[f"{symbol_lower}_close"]]
        if close_df.empty:
            close_df = df_close
        else:
            close_df = pd.merge(
                close_df, df_close, how="outer", left_index=True, right_index=True
            )
    close_df.ffill().sort_index(inplace=True)
    close_df.to_csv(output_file)
    return close_df


# --- Command-line interface using click ---
@click.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--timeframe",
    "-t",
    default="M1",
    type=click.Choice(["M1", "H1", "D1"]),
    help="Timeframe for OHLCV data (default: M1)",
)
@click.option("--start_date", "-sd", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end_date", "-ed", required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--output_file",
    "-o",
    default="merged_data.csv",
    help="Filename for the full merged dataset CSV",
)
@click.option(
    "--output_close",
    "-oc",
    default="close_prices.csv",
    help="Filename for the close prices CSV",
)
def main(symbols, timeframe, start_date, end_date, output_file, output_close):
    """
    Download Dukascopy candle data (always using the chosen timeframe) via duka for the given symbols,
    process and merge the data into a full OHLCV dataset, and extract a separate close price dataset.

    Symbols should be provided as positional arguments.
    """
    download_folder = "./dukadata"
    os.makedirs(download_folder, exist_ok=True)

    # Convert start_date and end_date from strings to date objects
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Download data for each symbol using the duka library (candle mode)
    for symbol in symbols:
        print(f"Downloading {symbol} data...")
        download_asset(symbol, start, end, timeframe, download_folder)

    # Merge downloaded CSV files into a full dataset
    merged_data = merge_datasets(download_folder, symbols, output_file)
    print(f"Full merged dataset saved to {output_file}")
    print(merged_data.head())

    # Extract close prices from each symbol and merge into a separate dataset
    close_data = extract_close_prices(download_folder, symbols, output_close)
    print(f"Close prices dataset saved to {output_close}")
    print(close_data.head())


if __name__ == "__main__":
    main()

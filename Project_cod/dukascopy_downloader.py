import pandas as pd
import click
import os
import re
from datetime import datetime, timedelta
from duka.app import app as duka_app
from duka.core.utils import TimeFrame


def download_ohlc(symbols, start_date, end_date, folder):
    """Download 1-minute OHLC data using duka library"""
    duka_app(
        symbols=symbols,
        start=start_date,
        end=end_date,
        threads=10,
        timeframe=TimeFrame.MINUTE1,
        folder=folder,
        header=True,
    )


def extract_date_from_filename(filename):
    # Look for dates in formats: YYYY-MM-DD, YYYYMMDD, YYYY_MM_DD
    date_formats = [
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        (r"\d{8}", "%Y%m%d"),
        (r"\d{4}_\d{2}_\d{2}", "%Y_%m_%d"),
    ]
    for pattern, date_format in date_formats:
        match = re.search(pattern, filename)
        if match:
            date_str = match.group()
            try:
                return datetime.strptime(date_str, date_format).date()
            except ValueError:
                continue
    raise ValueError(f"Date not found in filename: {filename}")


def process_symbol_data(file_path, symbol):
    """Process 1-minute OHLC data with bid/ask columns"""
    # Read raw data (adjust column names based on actual file structure)
    df = pd.read_csv(
        file_path,
        names=[
            "time",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_volume",
            "bid_volume",
        ],
        parse_dates=["time"],
    )
    df.set_index("time", inplace=True)

    # Add symbol prefix
    return df.add_prefix(f"{symbol}_")


def merge_datasets(folder, symbols, output_path):
    """Merge multiple assets into combined CSV files"""
    full_merged = pd.DataFrame()

    for symbol in symbols:
        # Find all symbol files sorted by date
        symbol_files = [f for f in os.listdir(folder) if f.startswith(symbol)]
        if not symbol_files:
            print(f"No files found for {symbol}, skipping...")
            continue

        # Process all files chronologically
        symbol_dfs = []
        for file in sorted(symbol_files, key=extract_date_from_filename):
            file_path = os.path.join(folder, file)
            try:
                df = process_symbol_data(file_path, symbol)
                symbol_dfs.append(df)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

        if symbol_dfs:
            merged_symbol = pd.concat(symbol_dfs).sort_index()
            full_merged = pd.merge(
                full_merged,
                merged_symbol,
                left_index=True,
                right_index=True,
                how="outer",
            )

    # Save outputs
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "combined_ohlc.csv")
    full_merged.sort_index().ffill().dropna().to_csv(output_file)
    print(f"Combined data saved to {output_file}")
    return full_merged


@click.command()
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    required=True,
    help="Instruments to download (e.g. EURUSD GBPUSD)",
)
@click.option("--start_date", "-sd", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end_date", "-ed", required=True, help="End date (YYYY-MM-DD)")
@click.option("--output_dir", "-o", default="./output", help="Output directory")
def main(symbols, start_date, end_date, output_dir):
    """Main execution function"""
    # Convert dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Create directories
    raw_dir = os.path.join(output_dir, "raw_1m")
    os.makedirs(raw_dir, exist_ok=True)

    # Download raw 1-minute data
    print("Downloading 1-minute data...")
    try:
        download_ohlc(symbols, start, end, raw_dir)
    except Exception as e:
        print(f"Download error: {str(e)}")

    # Process and merge
    print("Processing and merging data...")
    merged_df = merge_datasets(raw_dir, symbols, output_dir)

    print("Process completed successfully")


if __name__ == "__main__":
    main()

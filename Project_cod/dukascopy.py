import pandas as pd
import glob
from datetime import datetime
import numpy as np

# # Step 1: Specify the path and pattern
# path = "C:/Users/marko/OneDrive/Bureau/Marko_documents/Etudes/Master_1ère/2ème_semestre/ADA_2.0/Source_code_ADA/data/USSC2000/"  # Replace with your directory
# pattern = "USSC2000.IDXUSD_Candlestick_1_M_BID*.csv"  # Adjust based on your file naming

# # Step 2: Get the list of files and sort them
# file_list = glob.glob(path + pattern)
# file_list.sort()

# # Step 3: Read and concatenate the DataFrames
# df_list = [pd.read_csv(file) for file in file_list]

# # Step 4: Remove duplicate rows where 'Local time' overlaps between adjacent files
# for i in range(1, len(df_list)):
#     # Remove rows in df_list[i] that have 'Local time' present in df_list[i-1]
#     df_list[i] = df_list[i][
#         ~df_list[i]["Local time"].isin(df_list[i - 1]["Local time"])
#     ]
# eurusd = pd.concat(df_list, ignore_index=True)

# # # Step 5: Save the concatenated DataFrame
# # sp500_full.to_csv(path + "sp500_ask_full.csv", index=False)

# # print("Concatenation complete. The full SP500 dataset is saved as 'sp500_full.csv'.")


# import pandas as pd
# from dateutil import parser
# import pytz

# # # Step 1: Load SP500 data
# # sp500_full = pd.read_csv(
# #     r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_1ère\2ème_semestre\ADA_2.0\Source_code_ADA\data\bid_sp500\sp500_bid_full.csv"
# # )


# # Step 2: Adjust the date strings
# eurusd["Local time"] = eurusd["Local time"].str.replace("GMT", "").str.strip()

# # Step 3: Convert 'Local time' to datetime with specified format
# eurusd["Local time"] = pd.to_datetime(
#     eurusd["Local time"], format="%d.%m.%Y %H:%M:%S.%f %z"
# )

# eurusd["Local time"] = pd.to_datetime(eurusd["Local time"], utc=True)

# # Verify the conversion
# print("Data type after conversion:", eurusd["Local time"].dtype)
# print(type(eurusd["Local time"].iloc[0]))


# # Verify the data type
# print("Data type after parsing:", eurusd["Local time"].dtype)

# # Step 4: Ensure 'Local time' is in UTC (should be if timezone is handled)
# # If necessary, convert to UTC
# eurusd["Local time"] = eurusd["Local time"].dt.tz_convert("UTC")

# print(eurusd["Local time"].head())

# # sp500_full.to_csv("sp500_bid_utc.csv")


# # df = pd.read_csv(r"sp500_bid_utc.csv")

# eurusd["Local time"] = pd.to_datetime(eurusd["Local time"])

# eurusd["Local time"] = eurusd["Local time"].dt.tz_convert("UTC")

# print(eurusd["Local time"])

# eurusd["Local time"] = eurusd["Local time"].view("int64") // 10**6

# # eurusd = eurusd.drop(columns=["Unnamed: 0"])

# print(eurusd)

# eurusd.to_csv("ussc2000_bid_cleaned.csv")


def combine_ask_bid(ask_df: pd.DataFrame, bid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine ask and bid DataFrames to produce a normal price DataFrame.
    The resulting prices (Open, High, Low, Close) will be the midpoint
    of the ask and bid prices, and the volume will be the sum of ask and bid volumes.

    Parameters
    ----------
    ask_df : pd.DataFrame
        DataFrame containing ask prices. Expected columns:
        ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume'].

    bid_df : pd.DataFrame
        DataFrame containing bid prices. Expected columns:
        ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume'].

    Returns
    -------
    pd.DataFrame
        A DataFrame with combined normal prices. Columns:
        ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    # Merge on 'Local time'
    merged = pd.merge(
        ask_df, bid_df, on="Local time", suffixes=("_ask", "_bid"), how="inner"
    )

    # Compute midpoint prices
    merged["Open"] = (merged["Open_ask"] + merged["Open_bid"]) / 2
    merged["High"] = (merged["High_ask"] + merged["High_bid"]) / 2
    merged["Low"] = (merged["Low_ask"] + merged["Low_bid"]) / 2
    merged["Close"] = (merged["Close_ask"] + merged["Close_bid"]) / 2

    # Sum the volumes from ask and bid
    merged["Volume"] = merged["Volume_ask"] + merged["Volume_bid"]

    # Keep only the desired columns
    final_df = merged[["Local time", "Open", "High", "Low", "Close", "Volume"]]
    print(f"len bid : {len(bid_df)}")
    print(f"len ask : {len(ask_df)}")
    print(f"len final : {len(final_df)}")

    return final_df


print("Starting the data cleaning....")

# uk100_bid = pd.read_csv("data/UK100/uk100_bid_cleaned.csv")
# uk100_ask = pd.read_csv("data/UK100/uk100_ask_cleaned.csv")

# uk100_df = combine_ask_bid(uk100_ask, uk100_bid)
# uk100_df.to_csv("uk100_cleaned.csv")
# uk100_close_df = uk100_df[["Local time", "Close"]]
# uk100_close_df.to_csv("uk100_close_cleaned.csv")

# eurusd_bid = pd.read_csv("data/EURUSD/EURUSD_bid_cleaned.csv")
# eurusd_ask = pd.read_csv("data/EURUSD/EURUSD_ask_cleaned.csv")

# eurusd_df = combine_ask_bid(eurusd_ask, eurusd_bid)
# eurusd_df.to_csv("eurusd_cleaned.csv")
# eurusd_close_df = eurusd_df[["Local time", "Close"]]
# eurusd_close_df.to_csv("eurusd_close_cleaned.csv")

# gbpusd_bid = pd.read_csv("data/GBPUSD/GBPUSD_bid_cleaned.csv")
# gbpusd_ask = pd.read_csv("data/GBPUSD/GBPUSD_ask_cleaned.csv")

# gbpusd_df = combine_ask_bid(gbpusd_ask, gbpusd_bid)
# gbpusd_df.to_csv("gbpusd_cleaned.csv")
# gbpusd_close_df = gbpusd_df[["Local time", "Close"]]
# gbpusd_close_df.to_csv("gbpusd_close_cleaned.csv")

# xauusd_bid = pd.read_csv("data/Gold/XAUUSD_bid_cleaned.csv")
# xauusd_ask = pd.read_csv("data/Gold/XAUUSD_ask_cleaned.csv")

# xauusd_df = combine_ask_bid(xauusd_ask, xauusd_bid)
# xauusd_df.to_csv("xauusd_cleaned.csv")
# xauusd_close_df = xauusd_df[["Local time", "Close"]]
# xauusd_close_df.to_csv("xauusd_close_cleaned.csv")

# sp500_bid = pd.read_csv("data/SP500/sp500_bid_cleaned.csv")
# sp500_ask = pd.read_csv("data/SP500/sp500_ask_cleaned.csv")

# sp500_df = combine_ask_bid(sp500_ask, sp500_bid)
# sp500_df.to_csv("sp500_cleaned.csv")
# sp500_close_df = sp500_df[["Local time", "Close"]]
# sp500_close_df.to_csv("sp500_close_cleaned.csv")

# ustbond_bid = pd.read_csv("data/US_T-Bonds/USTBOND_bid_cleaned.csv")
# ustbond_ask = pd.read_csv("data/US_T-Bonds/USTBOND_ask_cleaned.csv")

# ustbond_df = combine_ask_bid(ustbond_ask, ustbond_bid)
# ustbond_df.to_csv("ustbond_cleaned.csv")
# ustbond_close_df = ustbond_df[["Local time", "Close"]]
# ustbond_close_df.to_csv("ustbond_close_cleaned.csv")

# xleusd_bid = pd.read_csv("data/XLE_US_USD/XLE_US_USD_bid_cleaned.csv")
# xleusd_ask = pd.read_csv("data/XLE_US_USD/XLE_US_USD_ask_cleaned.csv")

# xleusd_df = combine_ask_bid(xleusd_ask, xleusd_bid)
# xleusd_df.to_csv("xleusd_cleaned.csv")
# xleusd_close_df = xleusd_df[["Local time", "Close"]]
# xleusd_close_df.to_csv("xleusd_close_cleaned.csv")

# xlpusd_bid = pd.read_csv("data/XLP_US_USD/XLP_US_USD_bid_cleaned.csv")
# xlpusd_ask = pd.read_csv("data/XLP_US_USD/XLP_US_USD_ask_cleaned.csv")

# xlpusd_df = combine_ask_bid(xlpusd_ask, xlpusd_bid)
# xlpusd_df.to_csv("xlpusd_cleaned.csv")
# xlpusd_close_df = xlpusd_df[["Local time", "Close"]]
# xlpusd_close_df.to_csv("xlpusd_close_cleaned.csv")

# aus200_bid = pd.read_csv("data/AUS200/aus200_bid_cleaned.csv")
# aus200_ask = pd.read_csv("data/AUS200/aus200_ask_cleaned.csv")

# aus200_df = combine_ask_bid(aus200_ask, aus200_bid)
# aus200_df.to_csv("aus200_cleaned.csv")
# aus200_close_df = aus200_df[["Local time", "Close"]]
# aus200_close_df.to_csv("aus200_close_cleaned.csv")

# chi50_bid = pd.read_csv("data/CHI50/chi50_bid_cleaned.csv")
# chi50_ask = pd.read_csv("data/CHI50/chi50_ask_cleaned.csv")

# chi50_df = combine_ask_bid(chi50_ask, chi50_bid)
# chi50_df.to_csv("chi50_cleaned.csv")
# chi50_close_df = chi50_df[["Local time", "Close"]]
# chi50_close_df.to_csv("chi50_close_cleaned.csv")

# dollar_idx_bid = pd.read_csv("data/DOLLAR_IDX/dollar_idx_bid_cleaned.csv")
# dollar_idx_ask = pd.read_csv("data/DOLLAR_IDX/dollar_idx_ask_cleaned.csv")

# dollar_idx_df = combine_ask_bid(dollar_idx_ask, dollar_idx_bid)
# dollar_idx_df.to_csv("dollar_idx_cleaned.csv")
# dollar_idx_close_df = dollar_idx_df[["Local time", "Close"]]
# dollar_idx_close_df.to_csv("dollar_idx_close_cleaned.csv")

# eurbond_bid = pd.read_csv("data/EUR_Bonds/eurbond_bid_cleaned.csv")
# eurbond_ask = pd.read_csv("data/EUR_Bonds/eurbond_ask_cleaned.csv")

# eurbond_df = combine_ask_bid(eurbond_ask, eurbond_bid)
# eurbond_df.to_csv("eurbond_cleaned.csv")
# eurbond_close_df = eurbond_df[["Local time", "Close"]]
# eurbond_close_df.to_csv("eurbond_close_cleaned.csv")

# jpn225_bid = pd.read_csv("data/JPN225/jpn225_bid_cleaned.csv")
# jpn225_ask = pd.read_csv("data/JPN225/jpn225_ask_cleaned.csv")

# jpn225_df = combine_ask_bid(jpn225_ask, jpn225_bid)
# jpn225_df.to_csv("jpn225_cleaned.csv")
# jpn225_close_df = jpn225_df[["Local time", "Close"]]
# jpn225_close_df.to_csv("jpn225_close_cleaned.csv")

# ukbonds_bid = pd.read_csv("data/UK_Bonds/ukbonds_bid_cleaned.csv")
# ukbonds_ask = pd.read_csv("data/UK_Bonds/ukbonds_ask_cleaned.csv")

# ukbonds_df = combine_ask_bid(ukbonds_ask, ukbonds_bid)
# ukbonds_df.to_csv("ukbonds_cleaned.csv")
# ukbonds_close_df = ukbonds_df[["Local time", "Close"]]
# ukbonds_close_df.to_csv("ukbonds_close_cleaned.csv")

ussc2000_bid = pd.read_csv("data/USSC2000/ussc2000_bid_cleaned.csv")
ussc2000_ask = pd.read_csv("data/USSC2000/ussc2000_ask_cleaned.csv")

ussc2000_df = combine_ask_bid(ussc2000_ask, ussc2000_bid)
ussc2000_df.to_csv("ussc2000_cleaned.csv")
ussc2000_close_df = ussc2000_df[["Local time", "Close"]]
ussc2000_close_df.to_csv("ussc2000_close_cleaned.csv")

print("Data cleaning finished. ")

# df = pd.read_csv("sp500_bid_cleaned.csv")
# df2 = pd.read_csv("sp500_ask_cleaned.csv")

# df = df.drop(columns=["Unnamed: 0"])
# df2 = df2.drop(columns=["Unnamed: 0"])

# print(df)
# print()
# print(df2)

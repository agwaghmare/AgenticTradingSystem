"""Analyze backtest_trade_log.csv for suspicious patterns."""
import os
import sys
import pandas as pd

LOG = os.path.join(os.path.dirname(__file__), "backtest_trade_log.csv")


def main():
    if not os.path.exists(LOG):
        print(f"Missing {LOG}")
        sys.exit(1)

    df = pd.read_csv(LOG)
    print(f"Total exits: {len(df)}")
    print(f"Unique pairs with exits: {df['pair'].nunique()}")

    print("\n--- Exit reason distribution ---")
    print(df["reason"].value_counts().to_string())

    zscore_stops = df[df["reason"] == "zscore_stop"]
    print(f"\nZ-score stops: {len(zscore_stops)} (expect >0 if logic works)")

    dollar_stops = df[df["reason"] == "dollar_stop"]
    print(f"Dollar stops: {len(dollar_stops)}")

    # Per-pair win rate
    by_pair = df.groupby("pair").agg(
        exits=("pnl", "count"),
        wins=("pnl", lambda s: (s > 0).sum()),
        total_pnl=("pnl", "sum"),
    )
    by_pair["win_rate"] = by_pair["wins"] / by_pair["exits"]
    suspicious = by_pair[(by_pair["exits"] >= 2) & (by_pair["win_rate"] == 1.0)]
    print(f"\n--- Pairs with 100% win rate (>=2 exits) ---")
    if suspicious.empty:
        print("  None found")
    else:
        print(suspicious.sort_values("total_pnl", ascending=False).to_string())

    always_lose = by_pair[(by_pair["exits"] >= 2) & (by_pair["win_rate"] == 0.0)]
    print(f"\n--- Pairs with 0% win rate (>=2 exits) ---")
    if always_lose.empty:
        print("  None found")
    else:
        print(always_lose.sort_values("total_pnl").to_string())

    print("\n--- Largest single-trade P&L (check for outliers) ---")
    print(df.nlargest(5, "pnl")[["pair", "exit_date", "reason", "pnl"]].to_string(index=False))
    print(df.nsmallest(5, "pnl")[["pair", "exit_date", "reason", "pnl"]].to_string(index=False))

    # Dollar stop should imply adverse move - flag exits where dollar_stop but huge profit
    if len(dollar_stops):
        big_win_dollar = dollar_stops[dollar_stops["pnl"] > 200]
        print(f"\n--- Dollar-stop exits with P&L > $200 (unusual) ---")
        if big_win_dollar.empty:
            print("  None")
        else:
            print(big_win_dollar[["pair", "exit_date", "pnl", "exit_z"]].to_string(index=False))

    # Missing exit_z on target/stop rows
    missing_z = df[df["reason"].isin(["target", "zscore_stop"]) & df["exit_z"].isna()]
    print(f"\nTarget/zscore_stop rows missing exit_z: {len(missing_z)}")


if __name__ == "__main__":
    main()

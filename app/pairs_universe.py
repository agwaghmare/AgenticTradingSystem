"""Shared sector universe and pair lists for live agent and backtest."""
from itertools import combinations

SECTOR_UNIVERSE: dict[str, list[str]] = {
    "consumer_staples": ["KO", "PEP", "PG", "CL", "COST", "WMT", "KMB", "CLX", "GIS", "KHC"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL"],
    "payments": ["V", "MA", "PYPL", "FIS", "GPN"],
    "retail": ["HD", "LOW", "TGT", "WMT", "TJX", "ROST", "DG", "DLTR", "BBY"],
    "banks": ["JPM", "BAC", "GS", "MS", "WFC", "USB", "PNC", "TFC", "C", "COF"],
    "healthcare": ["JNJ", "PFE", "UNH", "CVS", "MRK", "ABBV", "BMY", "LLY", "CI", "HUM"],
    "semis": ["QCOM", "AVGO", "TXN", "ADI", "MU", "NXPI", "MCHP", "ON"],
    "airlines": ["DAL", "UAL", "AAL", "LUV", "ALK", "JBLU"],
    "telecom": ["VZ", "T", "TMUS"],
    "industrials": ["HON", "GE", "MMM", "EMR", "ITW", "ETN", "PH", "ROK"],
    "reits": ["O", "SPG", "PLD", "AMT", "EQIX", "PSA", "DLR", "WELL"],
    "insurance": ["TRV", "ALL", "PGR", "CB", "AIG", "MET", "PRU"],
    "asset_managers": ["BLK", "BX", "KKR", "APO", "TROW", "BEN"],
    "autos": ["GM", "F", "STLA"],
    "beverages_alcohol": ["STZ", "BF-B", "TAP"],
}


def generate_candidate_pairs(sector_universe: dict[str, list[str]] | None = None) -> list[tuple[str, str]]:
    universe = sector_universe or SECTOR_UNIVERSE
    pairs: list[tuple[str, str]] = []
    for tickers in universe.values():
        for a, b in combinations(sorted(set(tickers)), 2):
            pairs.append((a, b))
    return pairs


FULL_CANDIDATE_PAIRS = generate_candidate_pairs()

# Portfolio-validated pairs — FROZEN for paper trading (see app/strategy_params.py)
# WARNING: in-sample Sharpe is inflated by selection bias; validate OOS via live_trade table.
HIGH_CONVICTION_PAIRS: list[tuple[str, str]] = [
    ("TGT", "WMT"),
    ("DG", "TGT"),
    ("OXY", "PSX"),
    ("BBY", "DG"),
    ("CL", "CLX"),
    ("LOW", "TJX"),
    ("EMR", "ITW"),
    ("EOG", "PSX"),
    ("COF", "PNC"),
    ("BLK", "KKR"),
    ("SLB", "VLO"),
    ("ALL", "CB"),
    ("PNC", "TFC"),
    ("GE", "HON"),
    ("T", "TMUS"),
    ("PSX", "VLO"),
    ("HD", "TJX"),
    ("ADI", "MCHP"),
    ("CVX", "VLO"),
]

# Live agent uses filtered pairs; backtest/param_sweep use FULL_CANDIDATE_PAIRS
CANDIDATE_PAIRS = HIGH_CONVICTION_PAIRS

# Frozen liquid single-name universe — NOT optimized on backtest P&L (avoids selection bias)
FROZEN_SINGLE_STOCK_UNIVERSE: list[str] = sorted(
    {
        "KO", "PEP", "PG", "WMT", "XOM", "CVX", "JPM", "BAC", "GS", "UNH",
        "JNJ", "HD", "LOW", "TGT", "V", "MA", "QCOM", "AVGO", "VZ", "T",
        "GE", "HON", "COST", "MRK", "ABBV",
    }
)

SINGLE_STOCK_UNIVERSE = FROZEN_SINGLE_STOCK_UNIVERSE

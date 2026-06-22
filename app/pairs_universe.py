"""Shared sector universe and pair generation for live agent and backtest."""
from itertools import combinations

SECTOR_UNIVERSE: dict[str, list[str]] = {
    "consumer_staples": ["KO", "PEP", "PG", "CL", "COST", "WMT", "KMB", "CLX", "GIS", "KHC"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HES"],
    "payments": ["V", "MA", "PYPL", "FIS", "FI", "GPN"],
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


CANDIDATE_PAIRS = generate_candidate_pairs()

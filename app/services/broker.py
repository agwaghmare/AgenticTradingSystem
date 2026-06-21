import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from app.config import settings

logger = logging.getLogger("broker")


class BrokerClient:
    def __init__(self):
        self.client: TradingClient | None = None
        self._sim_positions: dict[str, dict] = {}
        if settings.alpaca_api_key and settings.alpaca_secret_key:
            self.client = TradingClient(
                settings.alpaca_api_key,
                settings.alpaca_secret_key,
                paper="paper" in settings.alpaca_base_url,
            )
        else:
            logger.warning("Alpaca credentials not configured; broker calls will use simulation mode")

    def get_account_nav(self) -> float:
        if self.client is None:
            return 100_000.0
        account = self.client.get_account()
        return float(account.equity)

    def get_open_positions(self) -> list[dict]:
        if self.client is None:
            return [
                {
                    "ticker": ticker,
                    "qty": pos["qty"],
                    "market_value": pos["market_value"],
                    "avg_entry_price": pos["avg_entry_price"],
                }
                for ticker, pos in self._sim_positions.items()
                if abs(pos["qty"]) > 1e-8
            ]
        positions = self.client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
            }
            for p in positions
        ]

    def place_market_order(self, ticker: str, qty: float, side: str) -> dict:
        if self.client is None:
            signed_qty = abs(qty) if side == "BUY" else -abs(qty)
            existing = self._sim_positions.get(ticker, {"qty": 0.0, "avg_entry_price": 0.0, "market_value": 0.0})
            old_qty = existing["qty"]
            new_qty = old_qty + signed_qty
            price = existing["avg_entry_price"] or 100.0
            if old_qty == 0 or (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
                total_cost = abs(old_qty) * existing["avg_entry_price"] + abs(signed_qty) * price
                total_shares = abs(old_qty) + abs(signed_qty)
                avg = total_cost / total_shares if total_shares else price
            else:
                avg = existing["avg_entry_price"]
            self._sim_positions[ticker] = {
                "qty": new_qty,
                "avg_entry_price": avg,
                "market_value": new_qty * avg,
            }
            logger.info("Sim order: %s %s %.4f shares of %s", side, ticker, abs(qty), ticker)
            return {
                "broker_order_id": "sim",
                "status": "SIMULATED",
                "ticker": ticker,
                "qty": qty,
                "side": side,
            }

        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=ticker, qty=abs(qty), side=order_side, time_in_force=TimeInForce.DAY
        )
        order = self.client.submit_order(order_data=request)
        return {
            "broker_order_id": str(order.id),
            "status": order.status,
            "ticker": ticker,
            "qty": qty,
            "side": side,
        }

    def flatten_all_positions(self):
        if self.client is None:
            logger.info("Simulation mode: flatten_all_positions")
            self._sim_positions.clear()
            return
        self.client.close_all_positions(cancel_orders=True)

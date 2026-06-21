from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from app.config import settings


class BrokerClient:
    def __init__(self):
        self.client = TradingClient(
            settings.alpaca_api_key, settings.alpaca_secret_key, paper="paper" in settings.alpaca_base_url
        )

    def get_account_nav(self) -> float:
        account = self.client.get_account()
        return float(account.equity)

    def get_open_positions(self) -> list[dict]:
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
        self.client.close_all_positions(cancel_orders=True)

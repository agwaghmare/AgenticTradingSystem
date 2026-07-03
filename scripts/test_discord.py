"""Send a test message to your Discord webhook."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import notifier
from app.config import settings

if __name__ == "__main__":
    if not settings.discord_webhook_url:
        print("DISCORD_WEBHOOK_URL not set in .env")
        raise SystemExit(1)
    notifier.notify_startup(100_000.0, "Test", 19, 25)
    print("Test message sent — check your Discord channel.")

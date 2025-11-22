import os
from dotenv import load_dotenv
import psutil
import time
from concurrent.futures import ThreadPoolExecutor
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()
SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN")
# print("SLACK_TOKEN:", SLACK_TOKEN)
SLACK_CHANNEL = "#ml-updates"

# Executor for async Slack notifications
slack_pool = ThreadPoolExecutor(max_workers=1)


def _send_slack_async(message: str):
    """Send Slack message in a separate thread (non-blocking)."""
    def _send():
        try:
            client = WebClient(token=SLACK_TOKEN)
            client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
        except SlackApiError as e:
            print("Slack error:", e.response.get("error"))

    slack_pool.submit(_send)   # Fire and forget


def wait_for_disk_space_gb(
    path="/",
    min_free_gb=10,
    check_interval=30,
    alert_once=True
):
    """
    Pauses training when free disk < min_free_gb.
    Slack alert is sent async so it never blocks the main thread.
    """
    slack_alert_sent = False

    while True:
        usage = psutil.disk_usage(path)
        free_gb = usage.free / (1024**3)

        if free_gb >= min_free_gb:
            return free_gb  # Continue training

        # Only send Slack alert once
        if alert_once and not slack_alert_sent:
            message = (
                f"<!channel> :warning: *DISK SPACE ALERT*\n"
                f"Free space is only *{free_gb:.2f} GB* on `{path}`.\n"
                f"Training *paused* until at least `{min_free_gb} GB` is free."
            )
            _send_slack_async(message)
            slack_alert_sent = True

        print(f"[PAUSE] Free space {free_gb:.2f} GB < {min_free_gb} GB. Waiting for cleanup...")
        time.sleep(check_interval)


# ----------------------------------------
# SIMPLE USAGE IN TRAINING LOOP
# ----------------------------------------
if __name__ == "__main__":

    for step in range(100):
        # Before saving checkpoint OR every N steps
        wait_for_disk_space_gb(
            path="/home/ml",
            min_free_gb=30,
            check_interval=20,
        )

        # Your training code...
        print("Training step:", step)

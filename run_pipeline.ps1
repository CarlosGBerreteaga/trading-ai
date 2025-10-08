# C:\Users\carlo\Documents\Portfolio Projects\trading-ai\run_pipeline.ps1
# Launches the trading pipeline with enhanced annual-hold strategy and ntfy notifications.

$env:NTFY_TOPIC = "trading-ai-spy-alerts"
$env:NTFY_TOKEN = ""           # Optional bearer token if topic is protected
$env:TWILIO_ACCOUNT_SID = ""   # Leave blank unless SMS via Twilio is needed
$env:TWILIO_AUTH_TOKEN = ""
$env:TWILIO_FROM_NUMBER = ""

& "C:\Users\carlo\Documents\Portfolio Projects\trading-ai\.venv\Scripts\python.exe" `
  "C:\Users\carlo\Documents\Portfolio Projects\trading-ai\src\pipeline.py" `
  --symbol SPY `
  --years 20 `
  --prob-threshold 0.55 `
  --min-hold-days 252 `
  --ntfy-topic $env:NTFY_TOPIC `
  --ntfy-limit 1 `
  --ntfy-token $env:NTFY_TOKEN

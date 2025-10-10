# Launch the trading pipeline from PowerShell by delegating to WSL and the shared `.venv-linux`.

$distribution = "Ubuntu"  # Set to your installed WSL distribution name.
$projectRootWin = "C:\Users\carlo\Documents\Portfolio Projects\trading-ai"
$projectRootWSL = "/mnt/c/Users/carlo/Documents/Portfolio Projects/trading-ai"
$portfolioPathWSL = "$projectRootWSL/portfolio.json"

$ntfyTopic = "trading-ai-spy-alerts"
$ntfyToken = ""           # Optional bearer token if topic is protected
$twilioSid = ""           # Leave blank unless SMS via Twilio is needed
$twilioToken = ""
$twilioFrom = ""
$notifyPhone = ""         # Destination phone number for SMS alerts (e.g., +15551234567)

$notifyArg = if ($notifyPhone) { "--notify-phone `"$notifyPhone`"" } else { "" }

$command = @"
export NTFY_TOPIC='$ntfyTopic';
export NTFY_TOKEN='$ntfyToken';
export TWILIO_ACCOUNT_SID='$twilioSid';
export TWILIO_AUTH_TOKEN='$twilioToken';
export TWILIO_FROM_NUMBER='$twilioFrom';
cd "$projectRootWSL";
source .venv-linux/bin/activate;
python src/pipeline.py --symbol SPY --years 20 --prob-threshold 0.55 --min-hold-days 252 --portfolio "$portfolioPathWSL" --portfolio-update --ntfy-topic "$ntfyTopic" --ntfy-limit 1 --ntfy-token "$ntfyToken" $notifyArg;
"@

wsl.exe -d $distribution --cd "$projectRootWSL" bash -lc "$command"

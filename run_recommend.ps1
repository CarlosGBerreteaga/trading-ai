# Run the S&P 500 recommendation sweep through WSL using the shared `.venv-linux`.

$distribution = "Ubuntu"
$projectRootWSL = "/mnt/c/Users/carlo/Documents/Portfolio Projects/trading-ai"
$outputWSL = "$projectRootWSL/analysis_recommendations.csv"
$cacheWSL = "$projectRootWSL/sp500_tickers.txt"

$command = @"
export PYTHONUNBUFFERED=1;
cd "$projectRootWSL";
source .venv-linux/bin/activate;
python src/recommend.py --years 20 --prob-threshold 0.55 --min-hold-days 252 --max-tickers 500 --top 25 --min-trading-days 756 --ticker-cache "$cacheWSL" --output "$outputWSL";
"@

wsl.exe -d $distribution --cd "$projectRootWSL" bash -lc "$command"

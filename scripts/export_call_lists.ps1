param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

if ($env:PYTHONPATH) {
    if (-not $env:PYTHONPATH.Contains("src")) {
        $env:PYTHONPATH = "src;$($env:PYTHONPATH)"
    }
} else {
    $env:PYTHONPATH = "src"
}

python -m icp.cli.export_call_lists @RemainingArgs

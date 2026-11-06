param(
    [ValidateSet('qrecc')][string]$Dataset = 'qrecc',
    [switch]$Offline,
    [switch]$Tests
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = $null
foreach ($candidatePath in @(
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    'G:\Anaconda3\python.exe',
    (Join-Path $env:USERPROFILE 'anaconda3\python.exe'),
    (Join-Path $env:USERPROFILE 'miniconda3\python.exe')
)) {
    if (Test-Path -LiteralPath $candidatePath -PathType Leaf) { $pythonPath = $candidatePath; break }
}
if (-not $pythonPath) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $pythonPath = $pythonCommand.Source }
}
if (-not $pythonPath) { throw 'Python 3.10+ was not found. Install Python or create .venv first.' }
Write-Host "Python: $pythonPath"
if ($Tests) {
    & $pythonPath -X utf8 -B -m unittest discover -s tests -v
    exit $LASTEXITCODE
}
if (-not $Offline) {
    & $pythonPath -X utf8 -B ensure_ollama.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Ollama could not be started; the app will clearly show offline mode.'
    }
}
$appMode = if ($Offline) { 'demo' } else { 'auto' }
Write-Host "Dataset: $Dataset / Mode: $appMode / URL: http://127.0.0.1:8001"
Write-Host 'The first hybrid index is prepared in the background. Keep this window open.'
& $pythonPath -u -X utf8 -B app.py --dataset $Dataset --mode $appMode --port 8001
exit $LASTEXITCODE

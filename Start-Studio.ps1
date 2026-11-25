$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path $python)) {
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 or newer and retry.' }
}
$marker = '.venv/qs-requirements.txt'
if (!(Test-Path $marker) -or ((Get-FileHash requirements.txt).Hash -ne (Get-FileHash $marker).Hash)) {
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    Copy-Item requirements.txt $marker
}
& $python start.py @args
exit $LASTEXITCODE

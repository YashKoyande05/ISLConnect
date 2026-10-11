Write-Host "Cleaning up old Python 3.14 environment..." -ForegroundColor Cyan
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force .venv
}

Write-Host "Searching for stable Python (3.10, 3.11, or 3.12)..." -ForegroundColor Cyan
$python_exe = "python" # default fallback
$stable_found = $false

# Check py launcher
$py_list = (py --list 2>&1)
if ($py_list -match "3\.11") { $python_exe = "py -3.11"; $stable_found = $true }
elseif ($py_list -match "3\.10") { $python_exe = "py -3.10"; $stable_found = $true }
elseif ($py_list -match "3\.12") { $python_exe = "py -3.12"; $stable_found = $true }

# Try to look manually based on AppData (you had 3.10 installed)
$appdata_py310 = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
if (!$stable_found -and (Test-Path $appdata_py310)) {
    $python_exe = $appdata_py310
    $stable_found = $true
}

if (!$stable_found) {
    Write-Host "Could not automatically detect Python 3.10 or 3.11!" -ForegroundColor Yellow
    Write-Host "Creating environment with default python. If it fails, please install Python 3.11." -ForegroundColor Yellow
}

Write-Host "Creating fresh virtual environment..." -ForegroundColor Cyan
Invoke-Expression "$python_exe -m venv .venv"

Write-Host "Activating environment and installing dependencies..." -ForegroundColor Cyan
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install roboflow

Write-Host "Starting the app..." -ForegroundColor Green
.venv\Scripts\python.exe app.py

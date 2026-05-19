<#
.SYNOPSIS
    Set up a Python virtual environment and download the last 5 years of KBO data.

.DESCRIPTION
    This script:
      1. Locates a suitable Python interpreter (python / python3).
      2. Creates a virtual environment at .venv in the repository root (or reuses
         one that already exists).
      3. Activates the virtual environment.
      4. Installs dependencies from requirements.txt.
      5. Runs scripts/download_kbo_data.py to fetch and save KBO data.

.PARAMETER OutputDir
    Directory to save the downloaded data files. Defaults to data/kbo.

.PARAMETER Format
    Output file format: csv (default) or parquet.

.PARAMETER StartYear
    First season to include. Defaults to (EndYear - 4).

.PARAMETER EndYear
    Last season to include. Defaults to (current year - 1).

.PARAMETER Verbose
    Pass -Verbose to enable debug-level logging in the Python script.

.EXAMPLE
    # Run with defaults (last 5 complete seasons, CSV, data/kbo output directory)
    .\scripts\setup_and_download.ps1

.EXAMPLE
    # Custom range, parquet format
    .\scripts\setup_and_download.ps1 -StartYear 2021 -EndYear 2025 -Format parquet -OutputDir data/kbo
#>

[CmdletBinding()]
param(
    [string]  $OutputDir  = 'data/kbo',
    [ValidateSet('csv', 'parquet')]
    [string]  $Format     = 'csv',
    [int]     $StartYear  = 0,
    [int]     $EndYear    = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helper: write a coloured status message
# ---------------------------------------------------------------------------
function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 1. Resolve repository root (parent of the scripts folder)
# ---------------------------------------------------------------------------
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptDir
$VenvDir    = Join-Path $RepoRoot '.venv'
$ReqFile    = Join-Path $RepoRoot 'requirements.txt'
$DownloadScript = Join-Path $ScriptDir 'download_kbo_data.py'

Write-Step "Repository root: $RepoRoot"

# ---------------------------------------------------------------------------
# 2. Find a working Python interpreter
# ---------------------------------------------------------------------------
Write-Step "Locating Python interpreter"

$PythonExe = $null
foreach ($candidate in @('python', 'python3')) {
    try {
        $ver = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Found: $candidate ($ver)"
            $PythonExe = $candidate
            break
        }
    } catch {
        # candidate not on PATH – continue
    }
}

if ($null -eq $PythonExe) {
    Write-Error "No Python interpreter found. Install Python 3.9+ and ensure it is on your PATH."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Create the virtual environment if it does not already exist
# ---------------------------------------------------------------------------
if (Test-Path (Join-Path $VenvDir 'pyvenv.cfg')) {
    Write-Step "Virtual environment already exists at $VenvDir – reusing it"
} else {
    Write-Step "Creating virtual environment at $VenvDir"
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
}

# ---------------------------------------------------------------------------
# 4. Resolve the venv's Python / pip executables (cross-platform)
# ---------------------------------------------------------------------------
if (($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows) {
    $VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
    $VenvPip    = Join-Path $VenvDir 'Scripts\pip.exe'
} else {
    $VenvPython = Join-Path $VenvDir 'bin/python'
    $VenvPip    = Join-Path $VenvDir 'bin/pip'
}

# ---------------------------------------------------------------------------
# 5. Install requirements
# ---------------------------------------------------------------------------
Write-Step "Installing requirements from $ReqFile"

& $VenvPip install --upgrade pip --quiet
& $VenvPip install -r $ReqFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install failed."
    exit 1
}

# ---------------------------------------------------------------------------
# 6. Build the argument list for the download script
# ---------------------------------------------------------------------------
Write-Step "Preparing download arguments"

$DownloadArgs = @(
    $DownloadScript,
    '--output-dir', $OutputDir,
    '--format',     $Format
)

if ($StartYear -gt 0) {
    $DownloadArgs += '--start-year', $StartYear
}

if ($EndYear -gt 0) {
    $DownloadArgs += '--end-year', $EndYear
}

if ($PSBoundParameters.ContainsKey('Verbose') -or $VerbosePreference -eq 'Continue') {
    $DownloadArgs += '--verbose'
}

# ---------------------------------------------------------------------------
# 7. Run the download script inside the venv
# ---------------------------------------------------------------------------
Write-Step "Downloading KBO data"
Write-Host "  Command: $VenvPython $DownloadArgs"

& $VenvPython @DownloadArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "KBO data download failed (exit code $LASTEXITCODE)."
    exit $LASTEXITCODE
}

Write-Host "`nDone. Data saved to: $OutputDir" -ForegroundColor Green

param([Parameter(Mandatory=$true)][string]$Root, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
# A .NET launcher may inherit PowerShell 7 module paths. Windows PowerShell
# 5.1 must load its own utility/archive modules, not Core-only manifests.
if ($PSVersionTable.PSEdition -eq 'Desktop') {
    $env:PSModulePath = [IO.Path]::Combine($PSHOME, 'Modules') + ';' + [IO.Path]::Combine($env:ProgramFiles, 'WindowsPowerShell', 'Modules')
    Import-Module ([IO.Path]::Combine($PSHOME, 'Modules', 'Microsoft.PowerShell.Utility', 'Microsoft.PowerShell.Utility.psd1')) -Force
    Import-Module ([IO.Path]::Combine($PSHOME, 'Modules', 'Microsoft.PowerShell.Archive', 'Microsoft.PowerShell.Archive.psd1')) -Force
}
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$Root = [System.IO.Path]::GetFullPath($Root)
$lockStream = $null
$transcribing = $false
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
    $logs = Join-Path $Root 'logs'
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $lockStream = [IO.File]::Open((Join-Path $Root '.runtime.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    Start-Transcript -Path (Join-Path $logs 'bootstrap.log') -Append | Out-Null
    $transcribing = $true
    Set-Location -LiteralPath $Root
    # Keep installer/native temporary files on the submission volume.
    $temporaryRoot = Join-Path $Root '.bootstrap/tmp'
    [IO.Directory]::CreateDirectory($temporaryRoot) | Out-Null
    $env:TEMP = $temporaryRoot
    $env:TMP = $temporaryRoot
    $config = Get-Content -LiteralPath (Join-Path $Root 'environment/bootstrap.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $env:PYTHONPATH = (Join-Path $Root 'src') + ';' + (Join-Path $Root '_core')
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    $python = if ($env:EATDET_PYTHON) { [IO.Path]::GetFullPath($env:EATDET_PYTHON) } else { Join-Path $Root '.venv/Scripts/python.exe' }
    $good = $false
    Write-Host '[1/5] Checking local .venv and native dependencies...'
    if (Test-Path -LiteralPath $python) {
        & $python -c "import torch,numpy,scipy,pandas,h5py,imufusion,threadpoolctl; assert hasattr(torch,'load')"
        $good = ($LASTEXITCODE -eq 0)
    }
    $lockHash = (Get-FileHash -LiteralPath (Join-Path $Root 'environment/requirements.lock') -Algorithm SHA256).Hash
    $marker = Join-Path $Root '.venv/.competition-lock'
    if (-not $env:EATDET_PYTHON -and (-not (Test-Path -LiteralPath $marker) -or (Get-Content -LiteralPath $marker -Raw -ErrorAction SilentlyContinue).Trim() -ne $lockHash)) { $good = $false }
    if (-not $good) {
        Write-Host '[2/5] Preparing local Python. First installation requires internet access.'
        $bootstrap = Join-Path $Root '.bootstrap'
        New-Item -ItemType Directory -Force -Path $bootstrap | Out-Null
        $uvDir = Join-Path $bootstrap ('uv-' + $config.uv_version)
        $uv = Join-Path $uvDir 'uv.exe'
        $archive = Join-Path $bootstrap ('uv-' + $config.uv_version + '.zip')
        if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $config.uv_sha256) {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $temporary = $archive + '.partial'
            $client = [Net.HttpWebRequest]::Create($config.uv_url)
            $client.Timeout = 60000
            $response = $client.GetResponse()
            $inputStream = $response.GetResponseStream()
            $outputStream = [IO.File]::Create($temporary)
            try {
                $buffer = New-Object byte[] 262144
                [long]$received = 0
                $timer = [Diagnostics.Stopwatch]::StartNew()
                while (($read = $inputStream.Read($buffer,0,$buffer.Length)) -gt 0) {
                    $outputStream.Write($buffer,0,$read); $received += $read
                    $status = ('{0:N1} MB / {1:N1} MB; {2:N1} MB/s' -f ($received/1MB),($response.ContentLength/1MB),($received/1MB/[Math]::Max(.1,$timer.Elapsed.TotalSeconds)))
                    $percent = if ($response.ContentLength -gt 0) { [Math]::Min(100,[int](100*$received/$response.ContentLength)) } else { 0 }
                    Write-Progress -Activity 'Downloading environment installer' -Status $status -PercentComplete $percent
                }
                Write-Progress -Activity 'Downloading environment installer' -Completed
            } finally { $outputStream.Dispose(); $inputStream.Dispose(); $response.Dispose() }
            if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $config.uv_sha256) { throw 'Installer download checksum failed; rerun to retry.' }
            Move-Item -LiteralPath $temporary -Destination $archive -Force
        }
        # Re-extract verified executable so an old/corrupt tool cannot be reused.
        Expand-Archive -LiteralPath $archive -DestinationPath $uvDir -Force
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $bootstrap 'python'
        $env:UV_CACHE_DIR = Join-Path $bootstrap 'cache'
        $env:UV_PYTHON_BIN_DIR = Join-Path $bootstrap 'bin'
        $env:UV_NO_PROGRESS = 'false'
        # exFAT cannot create uv's minor-version directory junction. The full
        # verified download remains usable; validate and address it explicitly.
        $basePython = Join-Path $env:UV_PYTHON_INSTALL_DIR ('cpython-' + $config.python + '-windows-x86_64-none/python.exe')
        function Test-CompetitionPython([string]$Candidate) {
            if (-not (Test-Path -LiteralPath $Candidate)) { return $false }
            & $Candidate -c "import sys,struct; assert sys.version_info[:3] == tuple(map(int,'$($config.python)'.split('.'))) and struct.calcsize('P')==8" *> $null
            return ($LASTEXITCODE -eq 0)
        }
        if (-not (Test-CompetitionPython $basePython)) {
            & $uv python install $config.python --no-bin --no-registry --no-config
            if ($LASTEXITCODE -ne 0) {
                if (-not (Test-CompetitionPython $basePython)) { throw 'Python installation failed and no validated interpreter is available.' }
                Write-Host 'Python download verified; using its explicit path because filesystem aliases are unavailable.'
            }
        }
        if (-not (Test-CompetitionPython $basePython)) { throw 'Downloaded Python failed version/architecture validation.' }
        # Repair packages in place; recreate only an invalid/missing interpreter.
        $interpreterGood = $false
        if (Test-Path -LiteralPath $python) {
            & $python -c "import sys,struct; assert sys.version_info[:3] == tuple(map(int,'$($config.python)'.split('.'))) and struct.calcsize('P')==8"
            $interpreterGood = ($LASTEXITCODE -eq 0)
        }
        if (-not $interpreterGood) {
            $venv = Join-Path $Root '.venv'
            if (Test-Path -LiteralPath $venv) {
                $backup = Join-Path $bootstrap ('invalid-venv-' + [Guid]::NewGuid().ToString('N'))
                Move-Item -LiteralPath $venv -Destination $backup
            }
            & $uv venv (Join-Path $Root '.venv') --python $basePython --no-python-downloads --link-mode copy --no-config
            if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
        }
        Write-Host '[3/5] Downloading and installing locked CPU dependencies (progress below)...'
        & $uv pip sync (Join-Path $Root 'environment/requirements.lock') --python $python --require-hashes --only-binary ':all:' --link-mode copy --config-file (Join-Path $Root 'environment/uv.toml')
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; see progress/log and rerun.' }
        & $python -c "import torch,numpy,scipy,pandas,h5py,imufusion,threadpoolctl; assert hasattr(torch,'load')"
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Dependency test failed; reinstalling locked packages once...'
            & $uv pip sync (Join-Path $Root 'environment/requirements.lock') --python $python --require-hashes --only-binary ':all:' --reinstall --link-mode copy --config-file (Join-Path $Root 'environment/uv.toml')
            if ($LASTEXITCODE -ne 0) { throw 'Dependency repair failed.' }
        }
        Set-Content -LiteralPath $marker -Value $lockHash -Encoding ASCII
    }
    Write-Host '[4/5] Testing trained weights and real model forward...'
    & $python (Join-Path $Root 'run.py') check
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path -LiteralPath $marker) { Remove-Item -LiteralPath $marker }
        throw 'Environment/model self-test failed. See logs/bootstrap.log.'
    }
    if (-not $CheckOnly) {
        Write-Host '[5/5] Processing all subjects from ACC_test_data...'
        & $python (Join-Path $Root 'run.py') predict --data-root (Join-Path $Root 'ACC_test_data') --output (Join-Path $Root 'results.csv') --work-dir (Join-Path $Root '.work/prepared') --device cpu
        if ($LASTEXITCODE -ne 0) { throw 'Inference failed. See logs/last_error.log; no new results were committed.' }
    }
    $exitCode = 0
} catch {
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    $exitCode = 1
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    if ($lockStream) { $lockStream.Dispose() }
}
exit $exitCode

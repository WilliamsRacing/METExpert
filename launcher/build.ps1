$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
& $compiler /nologo /target:exe /platform:x64 /optimize+ "/out:$root\Competition2.exe" (Join-Path $PSScriptRoot 'Competition2.cs')
if ($LASTEXITCODE -ne 0) { throw 'Executable compilation failed.' }

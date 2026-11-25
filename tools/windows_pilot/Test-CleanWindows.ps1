# Place this file next to the RC3 setup EXE on a Windows Sandbox host.
[CmdletBinding()]
param([switch]$Guest)
$ErrorActionPreference = 'Stop'
$file = 'AutoPyro-0.1.1rc3-Setup-x64.exe'
$expected = 'EEEE46EB36C8A5D43F4150B8A430E62C18F2A92428F017773B4F8A35C7531E90'
$exe = Join-Path $PSScriptRoot $file

if ($Guest) {
    $out = 'C:\PilotResults'
    try {
        if ((Get-FileHash -LiteralPath $exe).Hash -ne $expected) { throw 'Unexpected installer SHA-256' }
        $active = @([Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
            Where-Object { $_.OperationalStatus -eq 'Up' -and $_.NetworkInterfaceType -ne 'Loopback' })
        if ($active.Count) { throw 'Guest has an active network interface' }
        foreach ($key in 'HKLM:\SOFTWARE\Python','HKLM:\SOFTWARE\WOW6432Node\Python','HKCU:\SOFTWARE\Python') {
            if (Test-Path $key) { throw "Python already registered: $key" }
        }
        $existing = @(Get-Command py.exe,python.exe,python3.exe,pip.exe -All -CommandType Application -ErrorAction SilentlyContinue |
            Where-Object { $_.Source -notlike '*\Microsoft\WindowsApps\*' })
        if ($existing.Count) { throw 'Python already available on PATH' }
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        Remove-Item Env:PYTHONHOME,Env:PYTHONPATH,Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
        $app = Join-Path $env:LOCALAPPDATA 'Programs\AutoPyro\0.1.1rc3'
        if (Test-Path $app) { throw 'Expected a fresh installation' }
        $setup = Start-Process -FilePath $exe -ArgumentList '/SILENT','/CURRENTUSER','/SUPPRESSMSGBOXES','/NORESTART','/LOG=C:\PilotResults\setup.log' -Wait -PassThru
        if ($setup.ExitCode -ne 0) { throw "Setup exit code $($setup.ExitCode)" }
        $reports = @(Get-ChildItem "$app\checks\*\control\result.json")
        if ($reports.Count -ne 1) { throw 'Missing unique control report' }
        $report = Get-Content -Raw -LiteralPath $reports[0].FullName | ConvertFrom-Json
        if (!$report.passed -or $report.report_comparison.difference_count -ne 0) { throw 'Reference mismatch' }
        $envPython = Join-Path $app 'env\Scripts\python.exe'
        $base = & $envPython -I -c "import sys;print(sys.base_prefix)"
        if ($LASTEXITCODE -ne 0 -or $base -ne (Join-Path $app 'runtime')) { throw 'Unexpected Python runtime' }
        foreach ($name in 'result.json','comparison.json','report.pdf','report.docx') {
            Copy-Item -LiteralPath (Join-Path $reports[0].DirectoryName $name) -Destination $out
        }
        [ordered]@{ passed=$true; setup_sha256=$expected; windows=[Environment]::OSVersion.VersionString;
            active_network_interfaces=$active.Count; python_before_install=$existing.Count; runtime=$base;
            setup_exit_code=$setup.ExitCode } | ConvertTo-Json | Set-Content "$out\sandbox-result.json" -Encoding UTF8
    } catch {
        $_.ToString() | Set-Content "$out\failure.txt" -Encoding UTF8
        throw
    }
    return
}

$sandbox = Get-Command WindowsSandbox.exe -CommandType Application -ErrorAction SilentlyContinue
if (!$sandbox) { throw 'Run on a host with Windows Sandbox enabled.' }
if ((Get-FileHash -LiteralPath $exe).Hash -ne $expected) { throw 'Unexpected installer SHA-256' }
$run = Join-Path $PSScriptRoot ('clean-windows-' + [guid]::NewGuid().ToString('N'))
$inputDir = New-Item -ItemType Directory -Path (Join-Path $run 'input') -Force
$outputDir = New-Item -ItemType Directory -Path (Join-Path $run 'results')
Copy-Item -LiteralPath $exe,$PSCommandPath -Destination $inputDir.FullName
$inputXml = [Security.SecurityElement]::Escape($inputDir.FullName)
$outputXml = [Security.SecurityElement]::Escape($outputDir.FullName)
$config = Join-Path $run 'offline.wsb'
@"
<Configuration>
  <Networking>Disable</Networking>
  <ClipboardRedirection>Disable</ClipboardRedirection>
  <MappedFolders>
    <MappedFolder><HostFolder>$inputXml</HostFolder><SandboxFolder>C:\PilotInput</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>$outputXml</HostFolder><SandboxFolder>C:\PilotResults</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>
  </MappedFolders>
  <LogonCommand><Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\PilotInput\Test-CleanWindows.ps1 -Guest</Command></LogonCommand>
</Configuration>
"@ | Set-Content -LiteralPath $config -Encoding UTF8
Write-Host "Results: $($outputDir.FullName)"
Start-Process -FilePath $sandbox.Source -ArgumentList ('"' + $config + '"')
$deadline = (Get-Date).AddMinutes(30)
while ((Get-Date) -lt $deadline) {
    if (Test-Path "$outputDir\failure.txt") { throw (Get-Content -Raw "$outputDir\failure.txt") }
    if (Test-Path "$outputDir\sandbox-result.json") { Write-Host 'Clean Windows check passed.'; return }
    Start-Sleep -Seconds 2
}
throw "Timed out waiting for the guest. See $outputDir."

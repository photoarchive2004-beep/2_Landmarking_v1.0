param(
  [string]$Title = "Choose LOCALITIES base (folder with <locality>\\png)",
  [switch]$Silent
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms | Out-Null

$tool = Split-Path -Parent $PSScriptRoot        # ...\tools\2_Landmarking_v1.0
$root = Split-Path -Parent $tool                # ...\GM
$cfg  = Join-Path $tool "cfg"
$logs = Join-Path $tool "logs"
$last = Join-Path $cfg  "last_base.txt"
New-Item -ItemType Directory -Force -Path $cfg,$logs | Out-Null

# стартовый путь
$init = ""
if (Test-Path $last) {
  try { $init = Get-Content $last -Raw } catch {}
}
if (-not $init) {
  $def = Join-Path $root "photos"
  $init = (Test-Path $def) ? $def : $root
}

# диалог выбора папки
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = $Title
$dlg.SelectedPath = $init
$dlg.ShowNewFolderButton = $false
$res = $dlg.ShowDialog()
if ($res -ne [System.Windows.Forms.DialogResult]::OK) {
  if (-not $Silent) { Write-Host "[INFO] Localities base selection cancelled by user." }
  exit 1
}

$chosen = $dlg.SelectedPath

# запомним выбранную базу
[System.IO.File]::WriteAllText($last, $chosen, (New-Object System.Text.UTF8Encoding($false)))

# создаём/обновляем джанкшн ROOT\photos -> выбранная база локальностей
$rootPhotos = Join-Path $root "photos"

function New-Junction($Path,$Target){
  try {
    New-Item -ItemType Junction -Path $Path -Target $Target -Force | Out-Null
    return $true
  } catch {
    return $false
  }
}

if (Test-Path $rootPhotos) {
  $it = Get-Item $rootPhotos -Force
  if ($it.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    Remove-Item $rootPhotos -Force
  } else {
    $isEmpty = -not (Get-ChildItem $rootPhotos -Force -Recurse -ErrorAction SilentlyContinue |
                     Where-Object { -not $_.PSIsContainer } | Select-Object -First 1)
    if ($isEmpty) {
      Remove-Item $rootPhotos -Recurse -Force
    } else {
      $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
      Rename-Item $rootPhotos ($rootPhotos + "_backup_$stamp") -Force
    }
  }
}

if (-not (New-Junction -Path $rootPhotos -Target $chosen)) {
  cmd /c "mklink /J ""$rootPhotos"" ""$chosen""" | Out-Null
}

# логируем факт
$log = Join-Path $logs "choose_localities_last.log"
$lines = @(
  ("TS=" + (Get-Date)),
  ("CHOSEN=" + $chosen),
  ("ROOT_PHOTOS=" + $rootPhotos),
  ("EXISTS_PNG=" + (Test-Path (Join-Path $chosen '*\png\*.png')))
)
$lines | Out-File -FilePath $log -Encoding utf8

if (-not $Silent) {
  Write-Host "[OK] Localities base:" $chosen
  Write-Host "[OK] Junction:" $rootPhotos "->" $chosen
}
exit 0
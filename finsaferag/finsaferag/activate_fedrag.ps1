# Kich hoat moi truong conda fedrag tren Windows PowerShell
# Chay: .\activate_fedrag.ps1   HOAC   powershell -ExecutionPolicy Bypass -File .\activate_fedrag.ps1

$condaPath = "D:\Sukem\anacoda"
$condaExe = "$condaPath\Scripts\conda.exe"
$condaPs1 = "$condaPath\shell\condabin\Conda.ps1"

# Kiem tra conda co ton tai khong
if (Test-Path $condaExe) {
    Write-Host "Tim thay conda tai: $condaExe" -ForegroundColor Green
} elseif (Test-Path "$condaPath\condabin\conda.bat") {
    $condaExe = "$condaPath\condabin\conda.bat"
    Write-Host "Tim thay conda tai: $condaExe" -ForegroundColor Green
} else {
    Write-Host "LOI: Khong tim thay conda. Kiem tra duong dan Anaconda (hien tai: $condaPath)" -ForegroundColor Red
    exit 1
}

# Cach 1: Load Conda.ps1 (neu co) roi activate
if (Test-Path $condaPs1) {
    Write-Host "Dang load Conda.ps1..." -ForegroundColor Cyan
    & $condaPs1
    conda activate fedrag
    Write-Host "Da kich hoat env: fedrag" -ForegroundColor Green
    Write-Host "Chay tiep lenh cua ban trong cung cua so nay." -ForegroundColor Yellow
    return
}

# Cach 2: Goi conda.exe truc tiep de activate (chi hieu qua trong session hien tai)
Write-Host "Dang kich hoat fedrag bang conda.exe..." -ForegroundColor Cyan
& $condaExe activate fedrag
if ($LASTEXITCODE -eq 0) {
    Write-Host "Da kich hoat env: fedrag" -ForegroundColor Green
} else {
    Write-Host "Thu chay:  $condaExe init powershell  roi dong mo lai PowerShell, sau do: conda activate fedrag" -ForegroundColor Yellow
}

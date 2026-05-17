$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:JUPYTER_CONFIG_DIR = Join-Path $root ".jupyter_config"
$env:JUPYTER_RUNTIME_DIR = Join-Path $root ".jupyter_runtime"
$env:IPYTHONDIR = Join-Path $root ".ipython_local"
$env:LOKY_MAX_CPU_COUNT = "4"

New-Item -ItemType Directory -Force -Path $env:JUPYTER_CONFIG_DIR, $env:JUPYTER_RUNTIME_DIR, $env:IPYTHONDIR | Out-Null

python -m jupyter nbconvert --to notebook --execute notebooks\06_week5_representation_dimensionality.ipynb --inplace --ExecutePreprocessor.timeout=1200
python -m jupyter nbconvert --to notebook --execute notebooks\07_week7_clustering_validation.ipynb --inplace --ExecutePreprocessor.timeout=1200

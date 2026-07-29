param(
    [switch]$RefreshEuropePmcScreen,
    [int]$MaxResults = 80
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

if ($RefreshEuropePmcScreen) {
    python scripts/publication/screen_external_phenotype_sources.py --max-results $MaxResults
}

python scripts/publication/build_external_validation_release.py
python scripts/publication/evaluate_external_phenotype_cohort.py `
    results/tpc_candidate_resource/external_validation/prelocked_external_phenotype_cohort.tsv `
    --bootstrap 10000

Write-Host "External-validation release assets regenerated."

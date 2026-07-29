<#!
.SYNOPSIS
Rebuild the frozen candidate-audit resource without retraining a model.

.DESCRIPTION
This command runs the release audit, component robustness calculation, DIAMOND
closest-labelled-homolog screen, evidence-card preparation, zero-overlap audit
and Figure 6 generation. It does not evaluate an external phenotype cohort;
that step is intentionally separate and requires a curator-locked input table.
#>

[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $root
try {
    & $Python "scripts\publication\build_tpc_candidate_registry.py"
    & $Python "scripts\publication\compute_candidate_ensemble_stability.py"
    & $Python "scripts\publication\assess_candidate_homology.py"
    & $Python "scripts\publication\prepare_core_candidate_evidence_cards.py"
    & $Python "scripts\publication\audit_candidate_exclusions.py"
    & $Python "scripts\manuscript\generate_audited_candidate_resource_figure.py"
}
finally {
    Pop-Location
}

Write-Host "Candidate audit pipeline completed."

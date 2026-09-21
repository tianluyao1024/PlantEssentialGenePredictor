# PlantEGP NAR-oriented web server entry point

The public NAR-oriented interface is `webapp/app.py`. `webapp/nar_app.py` is a
compatibility launcher for existing deployment commands. The prior interface is
retained locally as `webapp/legacy_app.py` for audit and rollback; it is not
the public NAR entry point.

## User workflow

PlantEGP provides two one-click worked examples (Arabidopsis and rice), plus
private prediction jobs from protein FASTA. GO, PPI and expression annotations
are optional; the server selects the compatible released feature profile from
the uploaded files. Domain annotation is not accepted because no deployed
profile uses it. The public interface supports the two species represented by
the released models: Arabidopsis thaliana and Oryza sativa.

Each submitted job receives an unguessable private result URL and has queued,
running, completed and failed states. A shared link grants access to its
result. Scores are prioritization scores for experimental follow-up under the
original study's severe loss-of-function evidence definition. They are not
calibrated probabilities of gene essentiality and do not establish phenotype
experimentally.

Completed-result views include score distributions, within-query rank
percentiles, priority bands, profile/input coverage, homology context where
available, a gene-level evidence view, and TSV, CSV, HTML, SVG and ZIP exports.
The two worked examples also show species-versus-joint model agreement. These
examples test reproducibility rather than independent biological validation.

## Privacy and retention

Jobs are stored under a private, token-named directory. Raw uploads and
intermediate files for terminal jobs are removed after 24 hours; reports remain
for seven days and then the terminal job directory is removed. Queued and
running jobs are excluded from automatic deletion. The app never lists job
directories to users and does not display server filesystem paths.

## Deployment assets

This Git repository intentionally excludes large model bundles, PLM weights,
processed matrices and private jobs. Obtain released model/data assets from the
project's versioned Zenodo record, configure `PLANT_EG_PLM_WEIGHTS` and, if GO
profiles are enabled, `PLANT_EG_GO_OBO`. Build or deploy the `NAR_webserver_20260919/examples`
and `webapp_data/nar_reference` assets from
`NAR_webserver_20260919/build_aligned_examples.py` and
`NAR_webserver_20260919/build_homology_resources.py`, or from a versioned
deployment package. Do not expose `webapp_data/jobs` as a static
directory.

For the existing NATAPP deployment, run the NAR entry point only on the
PlantEGP-dedicated loopback port. The companion ProtoEssAtlas service has a
separate upstream and must not be stopped or reconfigured by this deployment.

## Verification

Run:

```powershell
D:\Python\Python311\python.exe scripts\reproducibility\test_nar_preview.py
```

The test suite covers job-token validation, input validation, private reports,
queue admission, retention cleanup, recovery liveness, worked-result rendering
and optional homology-search failure handling. Before publication, perform a
fresh public-browser test of both worked examples and every download action.

## License and support

The source code is released under the [MIT License](../LICENSE). Source code,
versioned data/model assets and citation information are linked from the
website landing page. For operational support, use the contact address shown in
the web interface.

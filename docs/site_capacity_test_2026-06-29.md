# Website capacity test, 2026-06-29

## Test target

The current Streamlit website was tested in two modes:

1. raw input validation using the bundled complete mini input templates;
2. real model prediction using released processed 6751-dimensional feature matrices.

No small third plant species with complete raw FASTA, GFF3, GO, PPI, expression
and domain files is currently bundled in the repository. Therefore, this test
does not demonstrate raw data to probability prediction for a third species.

## Browser-level checks

Local app URL:

```text
http://localhost:8501
```

Observed pages:

- Full-model prediction page loads.
- Raw data upload demo page loads.
- Raw data upload page displays the three model families:
  - Arabidopsis single-species model;
  - rice single-species model;
  - joint Arabidopsis-rice model.
- Raw data upload page displays core inputs:
  - protein FASTA;
  - CDS FASTA;
  - GFF3.
- Raw data upload page displays optional inputs:
  - GO annotation TSV;
  - PPI edge-list TSV;
  - expression matrix TSV;
  - domain annotation TSV.

## Raw input validation test

Input templates:

```text
docs/input_templates/cds_example.fasta
docs/input_templates/protein_example.fasta
docs/input_templates/gff3_minimal_template.gff3
docs/input_templates/go_annotation_template.tsv
docs/input_templates/ppi_edges_template.tsv
docs/input_templates/expression_matrix_template.tsv
docs/input_templates/domain_annotation_template.tsv
```

Results:

| File | Rows/records | Required format result |
|---|---:|---|
| cds_example.fasta | 2 records | passed |
| protein_example.fasta | 2 records | passed |
| gff3_minimal_template.gff3 | 4 non-comment rows | passed, first rows have at least 9 tab-separated columns |
| go_annotation_template.tsv | 3 rows | passed, required columns `gene_id`, `go_id` |
| ppi_edges_template.tsv | 2 rows | passed, required columns `gene_a`, `gene_b`, `score` |
| expression_matrix_template.tsv | 2 rows | passed, required first column `gene_id` |
| domain_annotation_template.tsv | 3 rows | passed, required columns `gene_id`, `domain_id`, `source` |

## Processed-feature prediction test

Commands:

```bash
python scripts/prediction/predict_from_processed_features.py \
  --features data/processed_features/rice_common6751_all_genes.npz \
  --model rice_single \
  --out webapp_data/jobs/site_capacity_test_rice_single.tsv

python scripts/prediction/predict_from_processed_features.py \
  --features data/processed_features/rice_common6751_all_genes.npz \
  --model joint \
  --out webapp_data/jobs/site_capacity_test_rice_joint.tsv

python scripts/prediction/predict_from_processed_features.py \
  --features data/processed_features/arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz \
  --model arabidopsis_single \
  --out webapp_data/jobs/site_capacity_test_ath_single.tsv
```

Results:

| Test | Genes | Predicted essential | Predicted non-essential | Probability min | Probability median | Probability max |
|---|---:|---:|---:|---:|---:|---:|
| Rice all genes, rice single model | 34,215 | 15,030 | 19,185 | 0.0006 | 0.2952 | 0.9991 |
| Rice all genes, joint model | 34,215 | 19,627 | 14,588 | 0.0017 | 0.5353 | 0.9993 |
| Arabidopsis unknown genes, Arabidopsis single model | 20,460 | 1,679 | 18,781 | 0.0257 | 0.1330 | 0.8191 |

Output files:

```text
webapp_data/jobs/site_capacity_test_rice_single.tsv
webapp_data/jobs/site_capacity_test_rice_joint.tsv
webapp_data/jobs/site_capacity_test_ath_single.tsv
```

## Problems found

1. Raw-data-to-prediction is not complete yet.
   The website validates raw FASTA/GFF3/GO/PPI/expression/domain files and
   explains feature extraction, but it does not yet execute the complete backend
   pipeline from raw files to 6751-dimensional matrix to prediction.

2. File upload could not be fully automated in the browser tool used for this
   test. The same parsing logic was tested on the bundled templates from the
   filesystem instead.

3. The processed-feature prediction path works, but full raw feature extraction
   is still missing for:
   - longest transcript reconciliation across CDS, protein and GFF3;
   - GO summary mapping;
   - PPI degree calculation from edge lists;
   - expression summary calculation from matrix;
   - domain count calculation;
   - PLM extraction;
   - feature concatenation and ordering;
   - calling the selected feature-profile model.

4. Arabidopsis single-model prediction produced many scikit-learn version
   warnings. The model was saved under scikit-learn 1.7.2 and tested under
   scikit-learn 1.8.0. Deployment should pin the training/runtime dependency
   versions.

5. LightGBM warnings appeared because prediction arrays do not include feature
   names. This did not stop prediction, but the deployment pipeline should either
   pass named feature frames where expected or suppress documented harmless
   warnings.

6. Default binary thresholds can produce high essential fractions in genome-wide
   rice prediction, especially for the joint model. For genome-wide usage, the
   probability ranking should be emphasized over interpreting the predicted
   essential fraction as a biological prevalence estimate.

7. Temporary Cloudflare trycloudflare URLs can expire. A stable domain plus a
   named Cloudflare Tunnel is needed for long-term public use.

## Conclusion

The website can currently:

- expose the intended model choices and raw-data upload templates;
- validate small raw input templates;
- run real prediction from already processed 6751-dimensional feature matrices;
- display and download known labels and released prediction tables.

The website cannot yet:

- take a new species raw data bundle and directly return predicted probabilities.

The next engineering task is to implement the raw feature-extraction backend and
connect it to the existing model prediction functions.

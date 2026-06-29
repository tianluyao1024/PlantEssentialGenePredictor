# Input feature preparation guide

All uploaded files must use the same stable gene ID in the first column or FASTA
header. Transcript IDs are allowed, but the final model input is one
protein-coding transcript per gene. If multiple transcripts are present, use the
longest CDS/protein-coding transcript and report that choice in the metadata.

## Model input tiers

| Tier | Required user input | Model intention |
|---|---|---|
| Basic sequence mode | CDS FASTA and protein FASTA | FASTA-derived sequence features plus PLM embeddings. This model must be trained separately before public sequence-only prediction is enabled. |
| Sequence + GO | CDS/protein FASTA and GO annotation table | Use sequence/PLM features plus functional annotation. |
| Sequence + PPI | CDS/protein FASTA and PPI edge or degree table | Use sequence/PLM features plus network degree features. |
| Sequence + expression | CDS/protein FASTA and expression matrix or expression summary | Use sequence/PLM features plus expression breadth/variation summaries. |
| Full feature mode | Processed `common6751` `.npz` file, or all raw sources needed to build it | Uses the manuscript full model. |

Do not treat missing GO, PPI or expression values as biological zeros unless the
source explicitly means zero. Missing annotation is different from absence of
function, interaction or expression.

## Common ID rules

- FASTA headers should start with the gene ID: `>Gene001`.
- All tables must contain a `gene_id` column.
- IDs are case-sensitive unless a species-specific mapper is documented.
- Duplicate `gene_id` values are not allowed in final feature tables.
- If raw files contain isoforms, provide a `transcript_id` column and keep only
  the longest protein-coding transcript per `gene_id`.

## Feature groups

### Sequence and protein composition

Input:

- `cds.fasta`
- `protein.fasta`

Processing:

- CDS length and protein length are counted after removing whitespace.
- GC, AT, GC skew, AT skew and GC3 are computed from CDS.
- A/C/G/T frequencies are computed from CDS.
- Amino-acid frequencies are computed from protein sequence.
- Amino-acid group frequencies are computed for hydrophobic, polar, positive,
  negative, small, aromatic and sulfur-containing amino acids.
- Protein molecular weight, GRAVY, isoelectric point, aromaticity and
  instability index are computed from protein sequence.

### Gene span and gene structure

Input:

- `annotation.gff3`

Processing:

- `gene_span_bp = end - start + 1` for the gene feature.
- Transcript/CDS records are linked through `ID` and `Parent` attributes.
- If multiple protein-coding transcripts exist, the longest CDS transcript is
  selected before feature extraction.

### GO annotation

Input:

- `go_annotation.tsv` with columns `gene_id` and `go_id`.

Processing:

- GO IDs are collapsed into the GO summary features used by the common model,
  including development, stress response, translation, ribosome, nucleic-acid
  binding and other curated GO groups.
- A GO feature is encoded as present/absent for each gene after propagating or
  mapping terms to the curated summary categories used in the model.

### PPI or STRING network features

Input options:

- edge table: `gene_a`, `gene_b`, `score`;
- or degree table: `gene_id`, `string_network_connections_400`,
  `string_network_connections_700`.

Processing:

- `string_network_connections_400`: number of interaction partners with score
  at least 400.
- `string_network_connections_700`: number of interaction partners with score
  at least 700.
- Self-loops are ignored and undirected duplicate edges are collapsed.

### Expression features

Input options:

- expression matrix: `gene_id`, followed by sample columns;
- or expression summary: `gene_id`, `median_expression`,
  `expression_variation`, `expression_breadth`, optional
  `expression_module_size`.

Processing:

- Median expression is computed across samples.
- Expression variation is computed as a normalized dispersion statistic across
  samples.
- Expression breadth is the fraction of samples where expression is above the
  selected detection threshold.
- Co-expression module size requires a precomputed module assignment or
  co-expression workflow.

### Paralog and homolog summaries

Input:

- whole-proteome FASTA and, for tandem duplication, GFF3.

Processing:

- Within-species paralogs are computed from all-vs-all protein similarity.
- Gene family size, singleton status, maximum paralog percentage identity and
  top paralog bitscore are summarized per gene.
- Tandem duplicate status requires genomic coordinates.
- Cross-species homolog features require reference proteomes and a documented
  homology search.

### Domain features

Input:

- `domain_annotation.tsv` with columns `gene_id`, `domain_id`, `source`.

Processing:

- `domain_number`: number of unique domains per gene.
- `pfam_domain_number`: number of unique Pfam domains per gene.

### PLM embeddings

Input:

- protein FASTA.

Processing:

- ESM2, ProtBERT and ProtT5 embeddings are extracted from amino-acid sequences.
- Special tokens and padding are excluded from pooling.
- Mean and max pooling are combined as in the manuscript feature pipeline.
- Protein-level embedding blocks are L2-normalized before model training.

## Processed `.npz` format

For full-model prediction, users may directly upload a processed `.npz` file:

- `X`: numeric feature matrix, `n_genes x 6751`;
- `gene_id`: gene identifiers;
- optional `transcript_id` or `sequence_id`;
- optional `feature_names`;
- optional `n_bio = 95`.

The first 95 columns must match `data/processed_features/common6751_feature_names.tsv`;
the remaining 6,656 columns must be ordered as ESM2, ProtBERT and ProtT5.

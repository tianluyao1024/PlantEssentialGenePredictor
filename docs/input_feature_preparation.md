# Raw input and feature preparation guide

The web tool is designed for raw biological data upload. Users do not need to
manually construct the final model feature matrix. The server validates input
files, checks gene IDs, extracts features, concatenates feature blocks in the
released schema and then applies the selected model.

All tabular files must be tab-separated (`.tsv`). Gene IDs are case-sensitive
and must be consistent across FASTA, GFF3, GO, PPI, expression and domain files.

## Model choices

The website exposes three main model families.

| Model family | Intended use |
|---|---|
| Arabidopsis single-species model | Prediction for Arabidopsis features using the Arabidopsis-trained model. |
| Rice single-species model | Prediction for rice features using the rice-trained model. |
| Joint Arabidopsis-rice model | General plant prioritization using the joint model. This model also has feature-profile variants for different available raw-data combinations. |

For the joint model, users may choose the feature profile that matches the files
they provide:

- sequence + PLM;
- sequence + PLM + GO;
- sequence + PLM + PPI;
- sequence + PLM + expression;
- sequence + PLM + GO + PPI;
- sequence + PLM + GO + expression;
- sequence + PLM + PPI + expression;
- sequence + PLM + GO + PPI + expression;
- advanced full uploaded-feature profile.

This avoids treating missing GO, PPI or expression annotations as true biological
zeros.

## Core files

These files are expected for the standard raw-data workflow.

### `protein.fasta`

Required for protein length, amino-acid composition, protein physicochemical
features and protein language model embeddings.

```text
>Gene001
MASLTVAAAGG
>Gene002
MKRAVLPLGG
```

### `cds.fasta`

Strongly recommended for CDS length, GC content, GC skew, GC3 and nucleotide
composition.

```text
>Gene001
ATGGCTTCTCTAACCGTTGCTGCTGCTGGTGGT
>Gene002
ATGAAACGTGCTGTTCTTCCGCTTGGTGGT
```

### `annotation.gff3`

Strongly recommended for gene span and transcript structure. If multiple
transcripts exist, the backend uses GFF3 `ID` and `Parent` relationships to
select one longest protein-coding transcript per gene.

```text
Chr1    source    gene    1000    3500    .    +    .    ID=Gene001
Chr1    source    mRNA    1000    3500    .    +    .    ID=Gene001.1;Parent=Gene001
Chr1    source    CDS     1100    1400    .    +    0    Parent=Gene001.1
```

## Optional annotation files

### GO annotation

File name suggestion: `go_annotation.tsv`

Required columns:

```text
gene_id    go_id
Gene001    GO:0009790
Gene001    GO:0006950
Gene002    GO:0006412
```

Processing:

- GO IDs are mapped to the curated GO summary groups used by the model.
- A GO summary feature is encoded per gene after term mapping.
- Missing GO annotation is treated as missing annotation, not as evidence of no
  function.

### PPI edge list

File name suggestion: `ppi_edges.tsv`

Required columns:

```text
gene_a    gene_b    score
Gene001   Gene002   850
Gene001   Gene003   420
Gene002   Gene004   760
```

Processing:

- The network is treated as undirected.
- Self-loops are removed.
- Duplicate undirected edges are collapsed.
- `string_network_connections_400` is the number of interaction partners with
  `score >= 400`.
- `string_network_connections_700` is the number of interaction partners with
  `score >= 700`.
- If users do not have confidence scores, they may set all scores to `1000`.

Only this edge-list format is supported for PPI upload.

### Expression matrix

File name suggestion: `expression_matrix.tsv`

Required format:

```text
gene_id    sample_1    sample_2    sample_3
Gene001    12.4        8.9         0.0
Gene002    0.0         0.2         4.1
Gene003    25.1        21.7        19.8
```

Processing:

- Rows are genes and columns are samples.
- TPM, FPKM or normalized counts are recommended.
- Raw read counts are not recommended unless users have already normalized them.
- The backend computes median expression, expression variation and expression
  breadth.
- `expression_module_size` is not computed in the basic web workflow unless a
  future co-expression module workflow is enabled.

Only matrix upload is supported for expression upload.

### Domain annotation

File name suggestion: `domain_annotation.tsv`

Required columns:

```text
gene_id    domain_id    source
Gene001    PF00069      Pfam
Gene001    PF07714      Pfam
Gene002    PF00001      Pfam
```

Processing:

- `domain_number` is the number of unique domains per gene.
- `pfam_domain_number` is the number of unique Pfam domains per gene.

## Feature extraction summary

| Feature group | Raw input | Processing |
|---|---|---|
| CDS composition | `cds.fasta` | CDS length, GC, AT, GC skew, AT skew, GC3 and nucleotide frequencies. |
| Protein composition | `protein.fasta` | Protein length, amino-acid frequencies and amino-acid group frequencies. |
| Physicochemical features | `protein.fasta` | Molecular weight, GRAVY, isoelectric point, aromaticity and instability index. |
| Gene structure | `annotation.gff3` | Gene span and transcript-to-gene mapping for longest transcript selection. |
| GO | `go_annotation.tsv` | GO IDs collapsed into curated GO summary categories. |
| PPI | `ppi_edges.tsv` | Degree features at score thresholds 400 and 700. |
| Expression | `expression_matrix.tsv` | Median expression, expression variation and expression breadth. |
| Domain | `domain_annotation.tsv` | Unique domain count and Pfam domain count. |
| Within-species paralog features | whole-proteome `protein.fasta` plus optional `annotation.gff3` | All-vs-all protein similarity and genomic position summaries. |
| PLM embeddings | `protein.fasta` | ESM2, ProtBERT and ProtT5 embeddings extracted from amino-acid sequences. |

The web workflow does not request an external homolog table.

## Processed `.npz` format

Advanced users may upload a processed `.npz` file directly:

- `X`: numeric feature matrix;
- `gene_id`: gene identifiers;
- optional `transcript_id` or `sequence_id`;
- optional `feature_names`;
- optional `n_bio`.

For the full 6751-dimensional model, the first 95 columns must match the
released biological feature order, followed by the 6656 PLM dimensions.

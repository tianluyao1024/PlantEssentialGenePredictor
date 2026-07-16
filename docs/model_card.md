# Model card

## Intended use

PlantEssentialGenePredictor is intended for genome-scale prioritization of plant essential-gene candidates. It should be used as a ranking and hypothesis generation tool, not as a replacement for experimental validation.

## Main models

| Model | Training labels | Feature set | Intended use |
|---|---|---|---|
| Arabidopsis single-species model | strict expanded Arabidopsis labels | common6751 | Arabidopsis predictions and within-species evaluation |
| Rice single-species model | strict rice essential labels plus Tos17 N4 nonessential labels | common6751 | Rice predictions and within-species evaluation |
| Joint Arabidopsis-rice model | Arabidopsis plus rice labels | common6751 | cross-species plant candidate prioritization |

## Deployable web profiles

The web app also includes joint-model feature-profile variants for user-uploaded raw files:

- sequence + PLM;
- sequence + PLM + GO;
- sequence + PLM + PPI;
- sequence + PLM + expression;
- sequence + PLM + GO + PPI;
- sequence + PLM + GO + expression;
- sequence + PLM + PPI + expression;
- sequence + PLM + GO + PPI + expression;
- advanced full uploaded-feature profile.

These profiles avoid forcing missing GO, PPI or expression blocks to zero.

## Feature space

The full common model uses 6751 features:

- 95 shared biological features;
- 2560 ESM2 embedding features;
- 2048 ProtBERT embedding features;
- 2048 ProtT5 embedding features.

The PLM embeddings are sequence-derived. GO, PPI, expression, domain and gene structure features require user-provided or database-derived annotations.

## Limitations

- Plant essentiality labels are incomplete and can be phenotype-stage dependent.
- Nonessential labels are less certain than essential labels because absence of severe phenotype is not proof of dispensability in all environments.
- GO and PPI features may reflect annotation density and phenotype-proximal curation bias.
- Predictions for poorly annotated non-model species should prefer the sequence/PLM profile or be interpreted cautiously.

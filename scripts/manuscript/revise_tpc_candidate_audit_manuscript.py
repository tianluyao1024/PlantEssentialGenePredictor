"""Create a The Plant Cell-oriented candidate-audit manuscript revision.

This script deliberately reports the audited candidate resource and the
prespecified independent-validation protocol, but it does not invent external
phenotype validation results. The source document remains unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


SOURCE = Path(r"C:\Users\tly\Downloads\Plant_essential_gene_prediction_manuscript_web_deployment_Nature_style.docx")
OUTPUT = Path(r"C:\Users\tly\Downloads\Plant_essential_gene_prediction_manuscript_ThePlantCell_candidate_audit_draft.docx")
FIGURE = Path(r"E:\PlantEssentialGenePredictor\results\manuscript_figures\figure6_audited_candidate_resource\Figure6_audited_candidate_resource.png")


def set_text(paragraph, text: str) -> None:
    """Replace a paragraph while retaining its paragraph style."""
    for run in paragraph.runs:
        run._element.getparent().remove(run._element)
    run = paragraph.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(10)


def insert_after(paragraph, text: str = "", style: str | None = None):
    new_p = paragraph._parent.add_paragraph(style=style)
    paragraph._p.addnext(new_p._p)
    if text:
        new_p.add_run(text)
    return new_p


def insert_before(paragraph, text: str = "", style: str | None = None):
    new_p = paragraph._parent.add_paragraph(style=style)
    paragraph._p.addprevious(new_p._p)
    if text:
        new_p.add_run(text)
    return new_p


def find_paragraph(document: Document, prefix: str):
    for paragraph in document.paragraphs:
        if paragraph.text.startswith(prefix):
            return paragraph
    raise ValueError(f"paragraph not found: {prefix}")


def add_heading_after(paragraph, text: str):
    p = insert_after(paragraph, style="Heading 2")
    p.add_run(text)
    return p


def add_body_after(paragraph, text: str):
    p = insert_after(paragraph, style="Normal")
    p.add_run(text)
    return p


def add_caption_after(paragraph, text: str):
    p = insert_after(paragraph, style="Caption")
    p.add_run(text)
    return p


def add_figure_after(paragraph, figure: Path):
    p = insert_after(paragraph, style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(figure), width=Inches(6.9))
    return p


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not FIGURE.exists():
        raise FileNotFoundError(FIGURE)
    doc = Document(SOURCE)

    set_text(
        doc.paragraphs[0],
        "A leakage-aware cross-species framework prioritizes plant essential-gene candidates in Arabidopsis and rice",
    )
    set_text(
        doc.paragraphs[2],
        "The Plant Cell-oriented manuscript draft with frozen model evaluation, an audited candidate resource and a prespecified independent-validation protocol.",
    )
    abstract = find_paragraph(doc, "Essential genes are central")
    set_text(
        abstract,
        "Essential genes support plant viability, development and fertility, yet genome-scale experimental essentiality maps remain incomplete in plants. Here we developed a leakage-aware framework that integrates conservative phenotype-label curation with a common 6,751-dimensional representation of Arabidopsis thaliana and Oryza sativa genes. The representation combines 95 biological features with ESM2, ProtBERT and ProtT5 protein-language-model embeddings. On fixed held-out test sets, species-specific models achieved area under the receiver-operating-characteristic curve (AUC) values of 0.9175 in Arabidopsis and 0.8812 in rice. A joint model gave numerically higher rice AUC but no significant paired DeLong improvement, and modestly lower Arabidopsis AUC. Feature ablation and homology-cluster evaluation showed complementary biological and sequence-representation signal while exposing annotation dependence and sequence-similarity effects. To separate discovery from label reuse, we audited all feature-covered genes against study labels, pseudo-labels and raw phenotype-record sources. The resulting release contains 17,522 Arabidopsis and 17,457 rice true-unknown candidates, alongside a curated core set of ten candidates per species. We provide the locked splits, model assets, code, candidate audit, a public prediction interface and a prespecified independent phenotype-validation protocol. The resource is intended for evidence-based candidate prioritization rather than replacement of experimental confirmation.",
    )

    web_heading = find_paragraph(doc, "A web-accessible predictor supports")
    set_text(web_heading, "Public prediction interface and reusable candidate resource")
    web_body = find_paragraph(doc, "For raw-data workflows")
    set_text(
        web_body,
        "The public interface supports a reproducible raw-data workflow. Users upload protein and CDS FASTA files and may add GFF3, GO annotation, PPI edge-list, expression-matrix and domain-annotation files through distinct validated upload fields. The system preserves gene identifiers, reports feature coverage, retains the longest protein-coding transcript per gene and selects only a released model compatible with the available input profile. Raw uploads are temporary; only a user-authorized final species-level prediction table is eligible for public caching.",
    )
    web_body_2 = find_paragraph(doc, "To make the released models accessible")
    set_text(
        web_body_2,
        "PlantEssentialGenePredictor is implemented as a public Streamlit application (https://plantessentialgene.com). It provides species-specific Arabidopsis and rice full models, a joint Arabidopsis-rice model and deployable feature-profile models for incomplete annotations. The web application is a delivery mechanism for frozen models and auditable outputs; it is not used as biological validation of model predictions.",
    )

    interpretation_caption = find_paragraph(doc, "Figure 5. Feature-interpretation summary")
    h1 = add_heading_after(interpretation_caption, "Audited genome-scale candidate release")
    b1 = add_body_after(
        h1,
        "Genome-scale prediction tables were reclassified before candidate nomination. Each feature-covered gene was assigned exactly one release status: known_label_used_in_study, pseudo_label_used_in_study, phenotype_recorded_but_excluded or true_unknown_candidate. A candidate could enter the true-unknown class only if it had zero overlap with the strict modeling labels, Arabidopsis pseudo-labels where applicable, and every reconciled raw phenotype-record source. The automated exclusion audit confirmed zero prohibited overlaps for all released core candidates.",
    )
    b2 = add_body_after(
        b1,
        "This procedure retained 17,522 true-unknown Arabidopsis genes and 17,457 true-unknown rice genes. We then nominated ten core candidates per species using concordant essential calls at each frozen model's validation-derived threshold and a probability of at least 0.50 in all three frozen species-specific, joint and annotation-light model outputs. Core candidates were additionally ranked by ensemble-component robustness and screened against the closest labelled homolog. For each species, the displayed panel deliberately includes five candidates with and five without a stringent labelled-essential homolog. These strata are intended to distinguish candidates supported by local sequence neighbourhoods from candidates that are not readily explained by a close labelled homolog.",
    )
    figure_p = add_figure_after(b2, FIGURE)
    fig_caption = add_caption_after(
        figure_p,
        "Figure 6. Audited genome-scale candidate resource. (a) Feature-covered genes were separated into study labels, study pseudo-labels, genes with phenotype records that were excluded from the primary labels and true-unknown candidates. (b,c) Agreement of frozen species-specific, joint and annotation-light prediction families for the ten Arabidopsis and ten rice core candidates. The dashed line marks probability 0.5 and is shown only as a visual reference. (d) Closest-labelled-homolog audit. A candidate was called homology-supported only when its closest study-labelled homolog met the predefined identity and coverage thresholds. These panels describe nomination evidence, not independent phenotype validation.",
    )
    h2 = add_heading_after(fig_caption, "Prespecified independent phenotype-validation framework")
    b3 = add_body_after(
        h2,
        "Candidate-level functional annotations, GO terms, PPI attributes and expression summaries can help formulate mechanistic hypotheses, but they are not treated as independent validation when the same source family contributed to the feature matrix. We therefore created a locked external-cohort schema that requires each external phenotype record to document its publication or database source, release date, experimental modality, phenotype stage, essentiality rule and evidence that it was not used in label curation, pseudo-labeling, feature encoding, model fitting or threshold selection.",
    )
    b4 = add_body_after(
        b3,
        "Quantitative external-cohort performance will be reported only after this independently curated cohort has been locked before inspection of prediction outcomes. The accompanying evaluator calculates AUC, AUPRC, sensitivity, specificity, precision and F1 with stratified bootstrap confidence intervals without retraining or reselecting thresholds. Accordingly, the current core candidates are discovery hypotheses with transparent model and homology evidence, not confirmed essential genes.",
    )

    discussion_first = find_paragraph(doc, "This study supports three main conclusions")
    set_text(
        discussion_first,
        "This study supports three main conclusions. First, plant essential-gene ranking is feasible when phenotype-derived labels are curated conservatively and evaluated on locked splits. Second, protein-language-model representations and interpretable biological features provide complementary signal, while ablations reveal important annotation dependence that must be reported rather than hidden. Third, genome-scale predictions must be separated from the labels and phenotype records used to build the model. The audited release therefore distinguishes true-unknown candidates from study labels, pseudo-labels and phenotype-recorded but excluded genes, and treats candidate nomination as hypothesis generation rather than biological confirmation.",
    )
    discussion_anchor = find_paragraph(doc, "A second limitation is annotation bias")
    d1 = add_body_after(
        discussion_anchor,
        "The candidate resource is organized around a biological working model in which low redundancy, conserved core molecular complexes and developmentally constrained programs jointly increase the probability of essentiality. The homology-supported and homology-isolated strata were retained to prevent this model from being reduced to simple nearest-neighbour annotation transfer. The present data do not justify a strong claim of zero-shot transfer across plant species: joint training altered the sensitivity-specificity trade-off but did not significantly improve AUC by paired testing.",
    )
    add_body_after(
        d1,
        "The next evidentiary step is an external phenotype cohort or targeted mutant validation that is genuinely independent of all model-development decisions. The released cohort template, automated independence checks and candidate evidence-card schema make this transition reproducible. In the interim, public predictions should be used to prioritize experiments, combine orthogonal evidence and guide resource allocation rather than to replace phenotypic assays.",
    )

    method_anchor = find_paragraph(doc, "ROC AUC and AUPRC")
    mh1 = add_heading_after(method_anchor, "Candidate release audit and core-candidate nomination")
    mb1 = add_body_after(
        mh1,
        "For every feature-covered gene, identifiers were normalized to the study gene namespace and compared with strict modeling-label tables, the Arabidopsis 0.60/0.40 pseudo-label table and each raw phenotype-record source used during data collection. Genes were assigned a mutually exclusive publication status in the following order: known_label_used_in_study, pseudo_label_used_in_study, phenotype_recorded_but_excluded, or true_unknown_candidate. Only the final class was eligible for the candidate resource. An automated audit reports candidate intersections with every prohibited source; the required intersection count is zero.",
    )
    mb2 = add_body_after(
        mb1,
        "For each true-unknown gene, frozen species-specific, joint and annotation-light predictors were evaluated without refitting. Core candidates were selected only when all three models produced an essential call at their locked validation-derived thresholds and each probability was at least 0.50; they were then ranked by component-model robustness and stratified by DIAMOND similarity to the closest study-labelled gene. Homology-supported candidates met both the predefined identity and query-coverage criteria against a labelled essential gene; homology-isolated candidates did not. This screen is a sequence-neighbourhood audit and is not an external validation test.",
    )
    mh2 = add_heading_after(mb2, "Independent phenotype-validation protocol")
    add_body_after(
        mh2,
        "The external-cohort evaluator accepts only phenotype records that include a stable source identifier, publication or release date, experiment type, developmental stage, binary label under a prespecified rule and an explicit independence declaration. It rejects records overlapping model labels or pseudo-labels and records derived from feature sources that directly encode the target phenotype. The cohort must be frozen before predictions are inspected. Continuous probabilities are then evaluated without retraining; thresholded metrics use the already locked validation-derived threshold and 95% confidence intervals are computed by stratified bootstrap resampling.",
    )

    availability = find_paragraph(doc, "Code, processed label tables")
    set_text(
        availability,
        "Code, processed label tables, fixed train/validation/test splits, feature lists, trained model assets, genome-scale prediction tables and figure-generation scripts are available at https://github.com/tianluyao1024/PlantEssentialGenePredictor. The archived release is available from Zenodo (https://doi.org/10.5281/zenodo.21387076). Before submission, the next versioned archive will include the audited candidate-release tables, core candidate evidence cards, external-validation template and the corresponding release manifest. The web interface is available at https://plantessentialgene.com.",
    )

    # Insert a concise supplementary-data list before the reference heading.
    reference_heading = find_paragraph(doc, "References")
    supplement_heading = insert_before(reference_heading, style="Heading 1")
    supplement_heading.add_run("Supplementary data release")
    s1 = add_body_after(
        supplement_heading,
        "Supplementary Data 1: audited Arabidopsis genome-scale prediction table with mutually exclusive publication status. Supplementary Data 2: audited rice genome-scale prediction table with mutually exclusive publication status. Supplementary Data 3: Arabidopsis and rice core-candidate evidence cards. Supplementary Data 4: automated candidate-exclusion audit. Supplementary Data 5: locked independent phenotype-validation cohort template and evaluator protocol.",
    )
    s1.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.core_properties.title = "Leakage-aware plant essential-gene candidate prioritization"
    doc.core_properties.subject = "The Plant Cell-oriented manuscript revision"
    doc.core_properties.comments = "Candidate audit and independent-validation protocol added; no external phenotype results are fabricated."
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

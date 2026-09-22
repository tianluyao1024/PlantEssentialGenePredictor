"""PlantEGP web server; run: streamlit run webapp/app.py"""
from __future__ import annotations

import io
import zipfile
import time
import pandas as pd
import streamlit as st
import altair as alt
from nar_style import apply_style, hero
from nar_jobs import FILES, PROCESSED, RAW, ROOT, job_path, read_state, reference, submit

PUBLIC_EXAMPLE_TOKEN = "PlantEGP_demo_Arabidopsis_00000000000000000"

st.set_page_config(page_title="PlantEGP", page_icon="🌱", layout="wide")
apply_style()
hero(compact="job" in st.query_params)
st.caption("Evidence-aware severe loss-of-function prioritization · [MIT license](https://github.com/tianluyao1024/PlantEssentialGenePredictor/blob/main/LICENSE)")
st.markdown(f"[View an instant Arabidopsis example output](?job={PUBLIC_EXAMPLE_TOKEN})")


@st.fragment(run_every=5)
def pending_status(token: str):
    state = read_state(token)
    if state['state'] not in {'queued', 'running'}:
        st.rerun()
    from nar_recovery import execution_health
    health=execution_health(token)
    if health == 'no_worker_detected':
        st.warning('No processing worker was detected for this pending job. It may have been interrupted. Your saved inputs are preserved; contact yyngmc@gmail.com with your private job link for recovery. Please do not submit repeated copies.')
    elif health == 'unknown':
        st.caption('Worker availability cannot currently be confirmed. The saved status is shown below.')
    elapsed = max(0, int(time.time() - state.get('submitted_unix', time.time())))
    st.info('Waiting for a compute slot' if state['state'] == 'queued' else 'Analysis in progress')
    st.caption(f'Time since submission: {elapsed // 60} min {elapsed % 60} sec · Status updates every 5 seconds while this page is open.')
    st.write('You may close this page and return using the private link. Jobs are computed one at a time; runtime depends on input size and model loading.')
    if st.button('Refresh status'):
        st.rerun()


def show_results(token: str):
    try:
        state = read_state(token)
    except (ValueError, FileNotFoundError):
        st.error("This private job link is invalid or unavailable.")
        return
    if state["state"] == "complete":
        st.success("Analysis complete · Your results are ready")
    elif state['state'] not in {'queued', 'running'}:
        st.info("Job status: " + state["state"])
    with st.expander("Your private result link · bookmark or share"):
        st.markdown(f"[Open this result](?job={token})")
        st.code("?job=" + token, language=None)
        st.caption("Bookmark the current page. Anyone given this link can view the results; no email is required.")
    if state["state"] in {"queued", "running"}:
        pending_status(token)
        return
    if state["state"] == "failed":
        st.error(state["error"])
        return
    path = job_path(token)
    if state.get('input_validation'):
        with st.expander('Input quality · annotation coverage'):
            st.dataframe(pd.DataFrame(state['input_validation']).T, width='stretch')
            partial=[name for name,detail in state['input_validation'].items() if detail.get('query_coverage_fraction',1)<1]
            if partial:
                st.warning('Some query genes lack records in: '+', '.join(partial)+'. Missing annotations are not evidence of absent function.')
    if state.get("execution_manifest"):
        with st.expander("Reproducibility · models and software versions"):
            st.json(state["execution_manifest"])
    out = pd.read_csv(path / "results.tsv", sep="\t")
    if state["kind"] == "demo" and "reference_single_species_score" not in out:
        st.error("This earlier development example used an older input snapshot. Start a new example to obtain scores aligned to the master release. The old files have been preserved for audit.")
        return
    st.subheader("Your results")
    a, b, c, d = st.columns(4)
    a.metric("Genes", len(out))
    b.metric("Median prioritization score", f"{out.score.median():.3f}")
    c.metric("Top-priority candidates", int((out.get("prioritization_band", pd.Series(dtype=str)) == "top-priority").sum()))
    d.metric("Input route", "Worked example" if state["kind"] == "demo" else "Protein FASTA" if state["profile"] == "sequence_plm" else "With annotations")
    st.caption("Higher scores prioritize follow-up experiments. They are not proof of lethality or calibrated probabilities; the supported scope is Arabidopsis and rice.")
    charts = st.columns(2) if "joint_model_score" in out else [st.container()]
    with charts[0]:
        st.markdown("#### Score distribution")
        chart = alt.Chart(out).mark_bar(color="#247c60", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("score:Q", bin=alt.Bin(step=0.1, extent=[0,1]), title="Prioritization score", scale=alt.Scale(domain=[0,1])),
            y=alt.Y("count():Q", title="Genes", axis=alt.Axis(tickMinStep=1)), tooltip=[alt.Tooltip("count():Q", title="Genes")]).properties(height=270)
        st.altair_chart(chart, width="stretch")
    if "joint_model_score" in out:
        with charts[1]:
            st.markdown("#### Model agreement")
            chart = alt.Chart(out).mark_circle(color="#ad793d", size=55, opacity=.75).encode(
                x=alt.X("single_species_score:Q", title="Species model", scale=alt.Scale(domain=[0,1])),
                y=alt.Y("joint_model_score:Q", title="Joint model", scale=alt.Scale(domain=[0,1])),
                tooltip=["gene_id", alt.Tooltip("single_species_score:Q", format=".3f"), alt.Tooltip("joint_model_score:Q", format=".3f")]).properties(height=270).interactive()
            st.altair_chart(chart, width="stretch")
        st.caption("Species and joint scores were recomputed from aligned example matrices and checked against the master release (absolute tolerance 1e-5). All reference_* fields are archived annotations, including annotation-light scores. Score spread compares only the two freshly run models; agreement is not independent validation.")
        ref = reference(state["species"])
        distribution = pd.DataFrame({"Current example": out.score.describe(), "Aligned species reference": ref.single_species_score.describe()})
        with st.expander("Query and reference distributions"):
            st.dataframe(distribution)
    else:
        st.caption("Only the selected raw-input profile was run. Cross-model agreement and reference rank are not established for user sequences. Where available, actual sequence alignment supplies labelled-protein matches; their labels are not assigned to the query.")
    if 'homology_search_status' in out and out.homology_search_status.eq('search_failed_prediction_preserved').any():
        st.warning('Prediction completed, but the optional protein similarity search failed. No homology conclusion can be drawn; the prediction and report remain available.')
    search = st.text_input("Filter gene identifiers")
    shown = out[out.gene_id.astype(str).str.contains(search, case=False, regex=False)] if search else out
    primary = [c for c in ["query_rank", "gene_id", "score", "query_rank_percentile", "prioritization_band", "joint_model_score", "score_spread", "reference_label_status", "biological_feature_nonmissing_fraction"] if c in shown]
    st.dataframe(shown[primary], hide_index=True, width="stretch", column_config={
        "query_rank":st.column_config.NumberColumn("Rank",format="%d"),
        "gene_id":"Gene", "score":st.column_config.ProgressColumn("Priority score",min_value=0,max_value=1,format="%.3f"),
        "query_rank_percentile":st.column_config.NumberColumn("Within-query percentile", format="%.1f%%"),
        "prioritization_band":"Priority band",
        "joint_model_score":st.column_config.NumberColumn("Joint score",format="%.3f"),
        "score_spread":st.column_config.NumberColumn("Score difference",format="%.3f"),
        "reference_label_status":"Reference evidence status"})
    with st.expander("Complete results · all evidence fields"):
        st.dataframe(shown, hide_index=True, width="stretch")
    with st.expander("Inspect one gene and its evidence"):
        gene = st.selectbox("Gene", out.gene_id.tolist())
        st.dataframe(out.set_index("gene_id").loc[gene].rename("Value").astype(str))
        if "nearest_labelled_homologue" in out:
            row = out.set_index("gene_id").loc[gene]
            if pd.notna(row.get("nearest_labelled_homologue")):
                st.info(f"Closest labelled protein: {row['nearest_labelled_homologue']} · Identity {row['nearest_labelled_identity_percent']:.1f}% · Query coverage {row['nearest_labelled_query_coverage_percent']:.1f}%")
            st.caption("DIAMOND search: E-value ≤ 1e-5; identity and both coverages ≥ 20%. Best returned hit by bit score. No hit does not establish a novel or nonessential gene. Exact sequence matches do not resolve identical paralogues.")
        st.caption("Evidence fields describe the frozen source registry. true_unknown_candidate means absent from that registry, not from all published literature. Empty homology fields mean no result is available.")
    st.markdown("#### Take your results with you")
    if (path / "score_distribution.svg").exists():
        st.download_button("Download score distribution · editable SVG", (path / "score_distribution.svg").read_bytes(), file_name="PlantEGP_score_distribution.svg", mime="image/svg+xml")
    for filename, column in zip(["results.tsv", "results.csv", "report.html", "report.zip"],st.columns(4)):
        with column:
            st.download_button({"results.tsv":"TSV table","results.csv":"CSV table","report.html":"HTML report","report.zip":"Complete package"}[filename], (path / filename).read_bytes(), file_name="PlantEGP_" + filename, width="stretch")


if "job" in st.query_params:
    show_results(st.query_params["job"])
    st.markdown("[Start another job](?)")
    st.stop()

predict_tab, help_tab = st.tabs(["Predict and explore", "Help, interpretation and privacy"])
with predict_tab:
    st.subheader("Try a complete worked example")
    st.write("Each button runs the released species and joint models on 100 bundled genes. These examples test the workflow; they are not validation cohorts.")
    for species, column in zip(["arabidopsis", "rice"], st.columns(2)):
        with column, st.container(border=True):
            species_name = "Arabidopsis" if species == "arabidopsis" else "Rice"
            st.markdown("### " + species_name)
            st.caption("100 curated reference candidates · species + joint models")
            if st.button("Try " + species_name + " demo", key=species, type="primary"):
                try:
                    token = submit("demo", species)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                st.query_params["job"] = token
                st.rerun()
            file = PROCESSED / f"{species}_100genes_common6751.npz"
            if file.exists():
                st.download_button("Download example input", file.read_bytes(), file_name=file.name, key=species + "_download")
    st.divider()
    st.subheader("Analyze your protein sequences")
    st.caption("Basic mode needs only protein FASTA. Advanced files are optional; the matching released model profile is detected automatically.")
    species = st.selectbox("Organism", ["arabidopsis", "rice"])
    protein = st.file_uploader("Protein FASTA", type=["fasta", "fa", "faa"], help="One unique protein per gene; 1-100 sequences per submission.")
    uploads = {"protein": protein.getvalue()} if protein else {}
    with st.expander("Advanced: add sequence and functional annotations"):
        st.write("The server selects the matching model profile automatically. Missing features use the model's fitted imputer; absent annotation does not establish absence of biological function.")
        for key, label, extensions in [
            ("cds", "CDS FASTA", ["fa", "fasta", "fna"]),
            ("gff3", "GFF3 annotation", ["gff", "gff3"]),
            ("go", "GO: gene_id, go_id", ["tsv"]),
            ("ppi", "PPI: gene_a, gene_b, score", ["tsv"]),
            ("expression", "Expression: gene_id followed by numeric sample columns", ["tsv"])]:
            item = st.file_uploader(label, type=extensions, key=key)
            if item:
                uploads[key] = item.getvalue()
    st.caption("Free web server · no account required · 1–100 proteins, at most 10,000 residues each; protein FASTA up to 10 MB and all files together up to 50 MB. Up to 3 jobs may run or wait; one computes at a time. Do not submit confidential data.")
    if st.button("Submit private prediction", disabled=not protein, type="primary"):
        try:
            token = submit("raw", species, uploads)
            st.query_params["job"] = token
            st.rerun()
        except (ValueError, UnicodeError) as exc:
            st.error(str(exc))
    if RAW.exists():
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in FILES.values():
                if (RAW / name).exists():
                    z.write(RAW / name, name)
        st.download_button("Download matched raw-input templates (Arabidopsis)", archive.getvalue(), "PlantEGP_raw_templates.zip")

with help_tab:
    st.markdown("""
### What the scores mean
Higher values prioritize candidates under the phenotype definitions in the original model release. They do not establish gene essentiality experimentally. Query rank is relative to the submitted batch. Reference percentile is available only for the bundled reference examples. Score spread is not a confidence interval.

### Input and output
Start with protein FASTA. Optional files must use matching gene identifiers and the exact column names shown in the Advanced panel. Protein-only input may be less informative when biological annotations are missing. The server automatically detects the matching released GO/PPI/expression profile. Domain annotation is not currently used by a deployable profile, so it is deliberately not accepted in the public form. The output table, gene detail view and downloadable HTML/TSV/CSV report include all available evidence. User-supplied gene IDs alone are never used to assert prior labels or homology.

### How to interpret a result
The prioritization score ranks a gene for experimental follow-up under the original severe loss-of-function evidence definition; it is not a probability of essentiality. Within-query percentile and priority band compare only genes in the submitted job. A top-priority band means the highest 1% of that submitted batch, and does not provide a biological threshold. Model agreement is displayed only for worked examples that ran two released models. Annotation completeness records supplied feature coverage, while a homology hit reports protein similarity and reference-label context only. Neither field proves a phenotype.

[View the interactive Arabidopsis example output](?job=PlantEGP_demo_Arabidopsis_00000000000000000).

### Evidence and applicability
The reference registry distinguishes study labels, pseudo-labels, excluded phenotype records and candidates absent from the audited sources. This release is trained on Arabidopsis and rice. A held-out source tests distribution shift; a worked example tests reproducibility. Neither is a prospective validation cohort. The historical external cohort contains 16 Arabidopsis genes (9 E, 7 NE). The current eligibility audit excludes eight (3 E, 5 NE) because of overlap with archived phenotype sources; eight (6 E, 2 NE) remain pending verification of feature independence and selection without model scores. Zero records are currently approved for quantitative independent validation. Source overlap does not by itself establish use in model training. No rice locked cohort or prospective accuracy claim is presented here.

### Privacy and availability
The server stores inputs and results on the host, accessible through an unguessable private link. Sharing the link grants access. Nothing is added to a public dataset. There is no registration or email requirement. The application adds no advertising or analytics scripts; Streamlit usage reporting is disabled in the project configuration.

The server stores inputs and intermediate files privately for up to 24 hours after a completed or failed job, then removes them. Downloadable reports remain available for up to 7 days after completion or failure, then the full job is removed. Jobs still queued or running are never removed automatically. Sharing a private result link grants access. An interrupted worker may require administrator recovery. For support or a deletion request, contact yyngmc@gmail.com and identify your job. Do not submit confidential data.
""")

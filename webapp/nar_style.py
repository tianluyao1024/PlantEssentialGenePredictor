"""Shared presentation layer for the PlantEGP web interface."""
import streamlit as st


def apply_style():
    st.markdown("""<style>
    :root {--ink:#183f37;--accent:#13745b;--paper:#f5f7f3;--muted:#60726b;}
    .stApp {background:var(--paper);color:var(--ink);}
    [data-testid="stHeader"] {background:transparent;}
    .block-container {max-width:1240px;padding-top:2.6rem;padding-bottom:3rem;}
    h1,h2,h3 {color:var(--ink);letter-spacing:-.035em;}
    h3 {font-size:1.3rem!important;font-weight:650!important;}
    p, label {color:#354e45;}
    [data-testid="stCaptionContainer"] p {color:var(--muted);font-size:.83rem;line-height:1.55;}
    .eg-hero {background:#123f34;border-radius:24px;padding:32px 38px;margin:0 0 24px;color:#fff;position:relative;overflow:hidden;}
    .eg-hero:after {content:'';position:absolute;right:-70px;top:-140px;width:420px;height:420px;border:1px solid #ffffff18;border-radius:50%;box-shadow:0 0 0 55px #ffffff06,0 0 0 110px #ffffff04;pointer-events:none;}
    .eg-brand {font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:#a8d3bb;margin-bottom:16px;}
    .eg-hero h1 {color:#fff;font-size:42px;line-height:1.12;margin:0 0 10px;letter-spacing:-.045em;}
    .eg-hero p {color:#d5e5da;max-width:660px;font-size:16px;line-height:1.65;margin:0;}
    .eg-compact {padding:22px 30px;margin-bottom:16px;}
    .eg-compact h1 {font-size:30px;}
    .eg-compact .eg-brand {margin-bottom:8px;}
    .eg-compact p,.eg-compact .eg-pills {display:none;}
    .eg-pills {display:flex;flex-wrap:wrap;gap:8px;margin-top:22px;}
    .eg-pills span {border:1px solid #ffffff30;padding:5px 12px;border-radius:20px;color:#e0eee4;font-size:11px;letter-spacing:.02em;}
    [data-testid="stMetric"] {background:white;border:1px solid #dde6de;border-radius:16px;padding:18px 22px;}
    [data-testid="stMetricValue"] {color:var(--ink);font-size:2rem;letter-spacing:-.04em;}
    [data-testid="stMetricLabel"] {color:var(--muted);}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px!important;}
    [data-testid="stExpander"] {background:#fff;border:1px solid #dde6de;border-radius:14px;}
    [data-testid="stFileUploader"] {background:#fff;border-radius:16px;padding:12px;}
    [data-testid="stFileUploaderDropzone"] {background:#f3f7f1;border:1px dashed #8eb59e;border-radius:12px;}
    .stButton button,.stDownloadButton button {border-radius:10px;min-height:42px;font-weight:550;}
    .stButton button[kind="primary"] {background:#167057;border-color:#167057;color:#fff;}
    .stButton button[kind="primary"]:hover {background:#0e5945;border-color:#0e5945;}
    button[data-baseweb="tab"] {font-size:14px;padding:10px 16px;}
    [data-baseweb="tab-highlight"] {background:#167057;}
    [data-testid="stAlert"] {border-radius:12px;font-size:.9rem;}
    [data-testid="stDataFrame"] {border:1px solid #dce5dd;border-radius:12px;overflow:hidden;}
    a {color:#167057!important;}
    .eg-kicker {font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#71847a;margin:22px 0 6px;}
    .eg-footer {margin-top:30px;border-top:1px solid #dce5dd;padding-top:16px;font-size:12px;color:#71847a;}
    @media(max-width:650px){.block-container{padding:1.3rem 1rem;}.eg-hero{padding:26px 23px;border-radius:18px;}.eg-hero h1{font-size:32px;}.eg-hero p{font-size:14px;}}
    </style>""", unsafe_allow_html=True)


def hero(compact=False):
    markup = """<div class="eg-hero">
    <div class="eg-brand">PLANT FUNCTIONAL GENOMICS · WEB SERVER</div>
    <h1>PlantEGP</h1>
    <p>From protein sequences to informed experiments.<br>Explore gene priorities with the evidence in view.</p>
    <div class="eg-pills"><span>Arabidopsis thaliana</span><span>Oryza sativa</span><span>No account required</span><span>Private analysis</span></div>
    </div>"""
    st.markdown(markup.replace('class="eg-hero"','class="eg-hero eg-compact"') if compact else markup, unsafe_allow_html=True)

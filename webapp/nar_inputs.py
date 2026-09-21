"""Validate text inputs before expensive embedding extraction."""
import io
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/feature_extraction"))
from raw_upload_to_profile_features import first_gene_id


def validate_annotations(uploads):
    proteins={first_gene_id(r.id) for r in SeqIO.parse(io.StringIO(uploads['protein'].decode('utf-8-sig')),'fasta')}
    summary={}
    schemas={'go':['gene_id','go_id'],'ppi':['gene_a','gene_b','score'],
             'expression':['gene_id'],'domain':['gene_id','domain_id','source']}
    for key,required in schemas.items():
        if key not in uploads:continue
        text=uploads[key].decode('utf-8-sig')
        header=text.splitlines()[0].split('\t') if text.strip() else []
        if len(header)!=len(set(header)):raise ValueError(f'{key}: duplicate column names.')
        try:
            frame=pd.read_csv(io.StringIO(text),sep='\t',dtype=str,keep_default_na=False)
        except (pd.errors.EmptyDataError,pd.errors.ParserError) as exc:
            raise ValueError(f'{key}: provide a well-formed tab-separated table.') from exc
        if frame.empty or not set(required)<=set(frame):raise ValueError(f'{key}: requires rows and columns {required}.')
        if frame[required].apply(lambda x:x.str.strip().eq('')).any().any():raise ValueError(f'{key}: required cells cannot be empty.')
        ids=set()
        for col in ['gene_a','gene_b'] if key=='ppi' else ['gene_id']:
            ids.update(frame[col].map(first_gene_id))
        matched=len(ids & proteins)
        if not matched:raise ValueError(f'{key}: no identifiers match the submitted proteins.')
        if key=='go' and not frame.go_id.str.fullmatch(r'GO:\d{7}').all():raise ValueError('GO identifiers must use GO: followed by seven digits.')
        numeric=['score'] if key=='ppi' else [c for c in frame if c!='gene_id'] if key=='expression' else []
        if key=='expression' and not numeric:raise ValueError('Expression table needs at least one numeric sample column.')
        if numeric:
            values=frame[numeric].apply(pd.to_numeric,errors='coerce').to_numpy(float)
            if not np.isfinite(values).all() or (values<0).any():raise ValueError(f'{key}: values must be finite, nonnegative numbers.')
        if key=='expression' and frame.gene_id.map(first_gene_id).duplicated().any():raise ValueError('Expression table must have one row per gene.')
        summary[key]={'rows':len(frame),'matched_query_genes':matched,'total_query_genes':len(proteins),
                      'query_coverage_fraction':matched/len(proteins),'unmatched_query_genes':sorted(proteins-ids)}
    if 'cds' in uploads:
        records=list(SeqIO.parse(io.StringIO(uploads['cds'].decode('utf-8-sig')),'fasta'))
        if not records or any(not r.seq or set(str(r.seq).upper())-set('ACGTUNRYKMSWBDHV') for r in records):raise ValueError('CDS: provide nonempty nucleotide FASTA sequences.')
        ids=[first_gene_id(r.id) for r in records]
        if len(ids)!=len(set(ids)):raise ValueError('CDS: submit one sequence per gene.')
        if not set(ids)&proteins:raise ValueError('CDS: no identifiers match the proteins.')
        summary['cds']={'records':len(records),'matched_query_genes':len(set(ids)&proteins)}
    if 'gff3' in uploads:
        rows=[r for r in uploads['gff3'].decode('utf-8-sig').splitlines() if r.strip() and not r.startswith('#')]
        if not rows:raise ValueError('GFF3 has no feature records.')
        for row in rows:
            fields=row.split('\t')
            if len(fields)!=9:raise ValueError('GFF3 requires exactly nine tab-separated columns.')
            try:
                if not 1<=int(fields[3])<=int(fields[4]):raise ValueError()
            except ValueError:raise ValueError('GFF3 coordinates must be positive ordered integers.')
        summary['gff3']={'records':len(rows)}
    return summary

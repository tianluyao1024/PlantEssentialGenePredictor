"""Self-contained report graphics generated from actual result values."""
import numpy as np


def score_distribution_svg(scores):
    counts, edges = np.histogram(np.asarray(scores, dtype=float), bins=np.linspace(0,1,11))
    peak = max(int(counts.max()),1)
    pieces=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 340" role="img" aria-label="Prioritization score distribution">',
            '<rect width="680" height="340" fill="#fff"/>',
            '<g font-family="Arial,sans-serif" fill="#183f37"><text x="45" y="28" font-size="18">Prioritization score distribution</text>']
    for i,count in enumerate(counts):
        height=220*count/peak;x=50+i*58;y=265-height
        pieces.append(f'<rect x="{x}" y="{y:.2f}" width="45" height="{height:.2f}" rx="3" fill="#247c60"/><text x="{x+22}" y="{y-7:.2f}" text-anchor="middle" font-size="12">{count}</text><text x="{x+22}" y="286" text-anchor="middle" font-size="11">{edges[i]:.1f}–{edges[i+1]:.1f}</text>')
    pieces.append('<text x="340" y="320" text-anchor="middle" font-size="12">Prioritization score · labels above bars show gene counts</text></g></svg>')
    return ''.join(pieces)

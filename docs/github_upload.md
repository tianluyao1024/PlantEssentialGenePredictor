# GitHub Upload Guide

The local release repository has already been initialized at:

```text
E:\PlantEssentialGenePredictor
```

Git LFS is enabled for large processed feature and model files:

```bash
git lfs install
git lfs track "*.npz"
git lfs track "*.joblib"
git lfs track "*.npy"
git lfs track "*.pt"
```

## Upload Steps

1. Create a public GitHub repository named:

```text
PlantEssentialGenePredictor
```

2. Do not initialize the GitHub repository with a README, license or `.gitignore`, because the local repository already has these files.

3. Add the remote and push:

```bash
cd /d E:\PlantEssentialGenePredictor
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/PlantEssentialGenePredictor.git
git push -u origin main
```

If the repository already exists as a remote:

```bash
git remote set-url origin https://github.com/<YOUR_GITHUB_USERNAME>/PlantEssentialGenePredictor.git
git push -u origin main
```

## Important Note About Large Files

This repository uses Git LFS. Before pushing, confirm that your GitHub account has enough LFS quota for the processed features and model files. If not, keep the code on GitHub and deposit large `.npz`/`.joblib` files on Zenodo, Figshare or OSF, then update `README.md` with the download DOI.


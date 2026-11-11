# TIPS Outcome Navigator

An English-language, interactive MT-GBM calculator for 365-day post-TIPS risk prediction.

The calculator estimates the probabilities of **rebleeding**, **overt hepatic encephalopathy**, and **all-cause mortality** from eight clinical inputs. This is an inference-only research prototype based on a trained shared-tree multi-task gradient boosting model.

## Required model bundle

**This repository template does not contain the fitted model.** Copy your already-exported `mtgbm_365.json.gz` to:

`models/mtgbm_365.json.gz`

The application will not provide predictions until the real fitted model is present.

## Local run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and upload **the contents of this directory**, preserving the folder structure. Include the fitted model under `models/`.
2. In https://share.streamlit.io, sign in using GitHub and select **Create app**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Under **Advanced settings**, choose Python **3.11** (or another supported and dependency-compatible version).
5. Deploy, then verify model readiness and check the same input against your local results.

## Privacy and publication

Do not publish patient-level CSV/XLSX data, identifiers, or private credentials. A model bundle uploaded to a **public** GitHub repository is also publicly downloadable. This calculator is intended for research use; outputs are based on fixed 365-day binary labels, not a competing-risk cumulative incidence model. It does not provide validated treatment recommendations.

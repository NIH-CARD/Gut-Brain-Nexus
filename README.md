# Gut-Brain Nexus: Mapping Multi-Modal Links to Neurodegeneration at Biobank Scale

`CARD ❤️ Open Science 😍`
[![Status - Pre-Print](https://img.shields.io/badge/Status-Pre--Print-0F4C92)](https://pubmed.ncbi.nlm.nih.gov/39371139/)

**Last Updated:** June 2025

## **Summary**

This repository contains the code, data workflows, and results associated with the manuscript:***Gut-Brain Nexus: Mapping Multi-Modal Links to Neurodegeneration at Biobank Scale.***

This study investigates:

1. The association between 155 endocrine, nutritional, metabolic, and digestive system disorders and AD / PD risk
2. The specificity of proteomic biomarkers for AD / PD in these subgroups
3. The development of multi-modal classification models integrating clinical, genetic, and proteomic data

## **Interactive Resource**

[https://gut-brain-nexus.streamlit.app](https://gut-brain-nexus.streamlit.app)

Explore:
* Feature importance
* Model interpretation
* Sample-level risk scores

## **Data Statement**

* UK Biobank data were used under application #33601 and accessed via the DNANexus analysis platform
* Data from the **FinnGen (R10)** and **SAIL** Biobanks were also leveraged
    * FinnGen Summary statistics and model outputs are available in the publication supplements

## **Citation**

If you use this repository or find it helpful for your research, please cite the corresponding manuscript (pre-print):

> **Gut-Brain Nexus: Mapping Multi-Modal Links to Neurodegeneration at Biobank Scale**
> *Shafieinouri MS, Hong S, Dadu A, Schuh A, Makarious MB, et al. medRxiv, 2024*
> https://pubmed.ncbi.nlm.nih.gov/39371139/

---

## **Workflow Diagram**

![image](https://hackmd.io/_uploads/ryUWlX5mgl.png)


---

## **Repository Orientation**

* The `finegray.ipynb` notebook contains the **survival analysis / competing risks modeling** pipeline.
* The `gut_brain_nexus.ipynb` notebook contains the main **multi-modal modeling** and **proteomic biomarker analyses**.

The `analyses/` directory includes all the analyses discussed in the manuscript or links to the respective GitHubs where analyses were followed.
```
this_repository
├── analyses
│   ├── finegray.ipynb
│   └── gut_brain_nexus.ipynb
├── LICENSE
├── README.md
└── streamlit
    └── ... [files related to hosting streamlit app]
```

---

## **Key Analyses**

1. **Survival analysis:**
   * Fine & Gray competing risks models
   * Time-stratified Cox regression models
   * Kaplan-Meier survival curves
2. **Polygenic Risk Scores (PRS):**
   * Calculation of PRS for **AD** and **PD**
   * Sensitivity analyses (APOE, LRRK2, GBA1 adjustments)
3. **Interaction modeling:**
   * GLM models to test interaction between PRS and ICD-10 codes
4. **Proteomics:**
   * Association of **Olink biomarkers** with AD / PD risk
   * Testing differential levels in cases with and without co-occurring conditions
5. **Multi-modal classification:**
   * Gradient Boosted Classifiers combining genetics, proteomics, demographics, and clinical features
   * Model interpretation with SHAP
   * Deployment of an **interactive Streamlit app**

---

## **Analysis Notebooks**

| Notebook                | Description                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `finegray.ipynb`        | Fine & Gray survival analysis pipeline on AD and PD cohorts                                                              |
| `gut_brain_nexus.ipynb` | Main multi-modal modeling pipeline, proteomics analysis, PRS modeling, interaction analyses, classifier training |

---

## **Software**

| Software             | Version(s) | URL                                                           | Notes                    |
| -------------------- | ---------- | ------------------------------------------------------------- | ------------------------ |
| Python               | 3.11.5       | [python.org](https://www.python.org/)                         | Main analysis language   |
| Pandas               | 2.2.3     | [pandas.pydata.org](https://pandas.pydata.org/)               | Data processing          |
| Scikit-learn         | 1.7.0    | [scikit-learn.org](https://scikit-learn.org/)                 | ML models                |
| XGBoost              | 3.0.2    | [xgboost.ai](https://xgboost.ai/)                             | ML models                |
| LightGBM             | 4.6.0     | [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/)   | Surrogate model for SHAP |
| SHAP                 | 0.44.0     | [shap.readthedocs.io](https://shap.readthedocs.io/)           | Model interpretation     |
| Lifelines            | 0.30.0     | [lifelines.readthedocs.io](https://lifelines.readthedocs.io/) | Survival analysis        |
| Matplotlib  | 3.6.3    |                    https://matplotlib.org/                              |                    Visualization      |
|  Seaborn |  0.12.2     |                 https://seaborn.pydata.org/                                 |                    Visualization      |



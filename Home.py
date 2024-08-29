import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

import warnings
warnings.filterwarnings("ignore")
import logging


logger = logging.getLogger()
logger.disabled = True

st.title("Gut-Brain Nexus: Mapping Multi-Modal Links to Neurodegeneration at Biobank Scale.")
st.header("Summary")
# st.subheader("Background")
st.write("Alzheimer’s disease (AD) and Parkinson’s disease (PD) are influenced by genetic and environmental factors. Using data from UK Biobank, SAIL Biobank, and FinnGen, we conducted an unbiased, population-scale study to: 1) Investigate how 155 endocrine, nutritional, metabolic, and digestive system disorders are associated with AD and PD risk prior to their diagnosis, considering known genetic influences; 2) Assess plasma biomarkers' specificity for AD or PD in individuals with these conditions; 3) Develop a multi-classification model integrating genetics, proteomics, and clinical data relevant to conditions affecting the gut-brain axis. Our findings show that certain disorders elevate AD and PD risk before AD and PD diagnosis including: insulin and non-insulin dependent diabetes mellitus, noninfective gastro-enteritis and colitis, functional intestinal disorders, and bacterial intestinal infections, among others. Polygenic risk scores revealed lower genetic predisposition to AD and PD in individuals with co-occurring disorders in the study categories, underscoring the importance of regulating the gut-brain axis to potentially prevent or delay the onset of neurodegenerative diseases. The proteomic profile of AD/PD cases was influenced by comorbid endocrine, nutritional, metabolic, and digestive systems conditions. Importantly, we developed multi-omics prediction models integrating clinical, genetic, and proteomic and demographic data, which perform better than a single paradigm approach  in disease classification accuracy for PD. This work aims to illuminate the intricate interplay between various physiological factors involved in the gut-brain axis and the development of AD and PD, providing a multifactorial systemic understanding that goes beyond traditional approaches.")

# st.subheader("Methods")
# st.write(".")

# st.subheader("Findings")
# st.write(".")

# st.subheader("Interpretation")
# st.write(".")

# st.subheader("Funding")
# st.write(".")

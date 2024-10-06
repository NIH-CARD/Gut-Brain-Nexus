import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import hashlib
# import plotly.express as px
# import plotly
import copy
import matplotlib.pyplot as plt
import gzip
import numpy
# import anndata as ad
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

def app():
    st.write("## Introduction")
    st.write(
        """
        The Shapley additive explanations (SHAP) approach was used to evaluate each feature’s influence in the ensemble learning. This approach, used in game theory, assigned an importance (Shapley) value to each feature to determine a player’s contribution to success. Shapley explanations enhance understanding by creating accurate explanations for each observation in a dataset. They bolster trust when the critical variables for specific records conform to human domain knowledge and reasonable expectations. 
        We used the one-vs-rest technique for multiclass classification. Based on that, we trained a separate binary classification model for each class.
        """
    )
    select_disease = st.selectbox("Select the disease  ", options=['ADRD', 'PD'])
    st.write('## Summary Plot')
    st.write("""Shows top-20 features that have the most significant impact on the classification model.""")
    if select_disease == 'ADRD':
        @st.cache_data
        def get_cache_data_ad():
            f = gzip.GzipFile("ad_shap_object_values_ms.npy.gz", "r")
            values = np.load(f)
            f.close()
            f = gzip.GzipFile("ad_shap_object_base_values_ms.npy.gz", "r")
            base_values = np.load(f)
            f.close()
            f = gzip.GzipFile("ad_shap_object_data_ms.npy.gz", "r")
            data = np.load(f)
            f.close()
            f = gzip.GzipFile("ad_shap_object_feature_names_ms.npy.gz", "r")
            feature_names = np.load(f)
            f.close()
            ad_shap_obj = shap.Explanation(values=np.array(values), base_values=base_values, data=data, feature_names=feature_names)
            with open("ad_top20_feature_list_ms.txt", 'r') as f:
                top20_features = f.read().split('\n')
            return ad_shap_obj, top20_features

        ad_shap_obj, top20_features = get_cache_data_ad()
    else:
        @st.cache_data
        def get_cache_data_pd():
            f = gzip.GzipFile("pd_shap_object_values_ms.npy.gz", "r")
            values = np.load(f)
            f.close()
            f = gzip.GzipFile("pd_shap_object_base_values_ms.npy.gz", "r")
            base_values = np.load(f)
            f.close()
            f = gzip.GzipFile("pd_shap_object_data_ms.npy.gz", "r")
            data = np.load(f)
            f.close()
            f = gzip.GzipFile("pd_shap_object_feature_names_ms.npy.gz", "r")
            feature_names = np.load(f)
            f.close()
            ad_shap_obj = shap.Explanation(values=np.array(values), base_values=base_values, data=data, feature_names=feature_names)
            with open("pd_top20_feature_list_ms.txt", 'r') as f:
                top20_features = f.read().split('\n')
            return ad_shap_obj, top20_features

        ad_shap_obj, top20_features = get_cache_data_pd()

    feature_names = ad_shap_obj.feature_names

    if True: # st.checkbox("Show Summary Plot"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('---')
            # temp = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns)
            # @st.cache_data
            def get_fig_bee():
                fig, ax = plt.subplots(figsize=(10,15))
                shap.plots.beeswarm(ad_shap_obj, show=False, max_display=20, plot_size=(10, 13))
                st.pyplot(fig)

            get_fig_bee()
            st.write('---')
        with col2:
            st.write('---')
            # @st.cache_data
            def get_fig_bar():
                fig, ax = plt.subplots(figsize=(10,15))
                shap.plots.bar(ad_shap_obj, show=False, max_display=20)
                st.pyplot(fig)

            get_fig_bar()
            st.write('---')
    
    st.write('## Dependence Plots')
    st.write("""We can observe the interaction effects of different features in for predictions. To help reveal these interactions dependence_plot automatically lists (top-3) potential features for coloring. 
    Furthermore, we can observe the relationship between features and SHAP values for prediction using the dependence plots, which compares the actual feature value (x-axis) against the SHAP score (y-axis). 
    It shows that the effect of feature values is not a simple relationship where increase in the feature value leads to consistent changes in model output but a complicated non-linear relationship.""")
    inds = []
    if True: # st.checkbox("Show Dependence Plots"):
        feature_name = st.selectbox('Select a feature for dependence plot', options=top20_features)
        
        @st.cache_data
        def get_inds(feature_name):
            try:
                return shap.utils.potential_interactions(ad_shap_obj[:, feature_name], ad_shap_obj)
            except:
                return []
        
        inds = get_inds(feature_name)
        if len(inds) <= 3:
            st.info("Select Another Feature")
        else:

            st.write('Top-3 Potential Interactions for ***{}***'.format(feature_name))

            @st.cache_data
            def get_interaction_plot(feature_name, inds):
                col3, col4, col5 = st.columns(3)
                with col3:
                    # fig, ax = plt.subplots(figsize=(8, 8))
                    shap.plots.scatter(ad_shap_obj[:, feature_name], color=ad_shap_obj[:, inds[0]])
                    # shap.dependence_plot(feature_name, np.array(ad_shap_obj.values), ad_shap_obj.data, interaction_index=list(ad_shap_obj.feature_names).index(list(ad_shap_obj.feature_names)[inds[0]]))
                    st.pyplot()
                with col4:
                    shap.plots.scatter(ad_shap_obj[:, feature_name], color=ad_shap_obj[:, inds[1]], show=False)
                    # shap.dependence_plot(feature_name, np.array(ad_shap_obj.values), ad_shap_obj.data, interaction_index=list(ad_shap_obj.feature_names).index(list(ad_shap_obj.feature_names)[inds[1]]))
                    st.pyplot()
                with col5:
                    shap.plots.scatter(ad_shap_obj[:, feature_name], color=ad_shap_obj[:, inds[2]], show=False)
                    # shap.dependence_plot(feature_name, np.array(ad_shap_obj.values), ad_shap_obj.data, interaction_index=list(ad_shap_obj.feature_names).index(list(ad_shap_obj.feature_names)[inds[2]]))
                    st.pyplot()

            get_interaction_plot(feature_name, inds)


app()
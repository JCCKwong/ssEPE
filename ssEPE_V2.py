"""
Title: Explainable artificial intelligence to predict the risk of side-specific extraprostatic extension in
pre-prostatectomy patients

Authors: Jethro CC. Kwong (1,2), Adree Khondker (3), Christopher Tran (3), Emily Evans (3), Amna Ali (4),
Munir Jamal (1), Thomas Short (1), Frank Papanikolaou (1), John R. Srigley (5), Andrew H. Feifer (1,4)

(1) Division of Urology, Department of Surgery, University of Toronto, Toronto, ON, Canada
(2) Temerty Centre for AI Research and Education in Medicine, University of Toronto, Toronto, Canada
(3) Temerty Faculty of Medicine, University of Toronto, Toronto, ON, Canada
(4) Institute for Better Health, Trillium Health Partners, Mississauga, ON, Canada
(5) Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, ON, Canada

This application predicts the risk of side-specific extraprostatic extension (ssEPE) using clinicopathological features
known prior to radical prostatectomy. The proposed use is for clinicians to input de-identified patient features and the
model will output a probability of ssEPE for the left and right lobe. This can be used in the context of patient
counselling and/or for tailoring surgical strategy (ie: nerve-sparing).
"""

# Import packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import streamlit as st
import shap
import joblib
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import ImageFont, ImageDraw, ImageOps
import SessionState


def main():
    st.title("Side-specific Extraprostatic Extension (ssEPE) Prediction Tool")
    st.sidebar.image("Logo.png", use_column_width=True)
    st.sidebar.header("Navigation")
    session_state = SessionState.get(button_id="", color_to_label={})
    PAGES = {
        "ssEPE Tool": full_app,
        "About": about,
        "Model Development": dev
    }
    page = st.sidebar.selectbox("",options=list(PAGES.keys()))
    PAGES[page](session_state)


# ssEPE Tool Page
def full_app(session_state):
    # Header text
    st.subheader("Instructions")
    st.markdown(
        """
    1. Enter patient values on the left
    1. Press submit button
    1. The ssEPE Tool will output the following:
        * Annotated prostate map showing location and severity of disease
        * Probability of ssEPE for the left and right prostatic lobe
        * Comparison of individual features to study population
    """
    )

    # Create 2 columns, one to show SHAP plots, one to show annotated prostate diagram
    col1, col2 = st.columns([1, 1.75])

    col1.subheader('Annotated Prostate')
    col1.write('Automatically updates based on user-entered values')
    col2.subheader('Model explanations')
    col2.write('The probability of ssEPE for each lobe is indicated in **bold**. \
    Each plot highlights which features have the greatest impact on the predicted probability of ssEPE')

    st.subheader('See how you compare with the study population')
    st.write('Each blue data point represents an individual case used to train this model, while '
                    'histograms on each plot show the distribution of values for that feature. The value that you '
                    'have inputted and its corresponding impact on probability of ssEPE is shown in **red**. This '
                    'helps you to visualize how your specific clinicopathological profile compares with the study '
                    'population to identify potential outliers.')

    colglobal, coly = st.columns([1, 1])
    colpsa, colmaxci, colphigh = st.columns([1, 1, 1])
    colpinv, colage, colz = st.columns([1, 1, 1])
    colside, colx = st.columns([1, 1])
    colbci, colmci, coltzci = st.columns([1, 1, 1])
    colbf, colwggg, colpc = st.columns([1, 1, 1])

    # Specify font size for annotated prostate diagram
    font = ImageFont.truetype('arial.ttf', 50)

    # Load saved items from Google Drive
    Model_location = st.secrets['Model']
    Feature_location = st.secrets['Feature']

    @st.cache(allow_output_mutation=True)
    def load_items():
        save_dest = Path('model')
        save_dest.mkdir(exist_ok=True)
        model_checkpoint = Path('model/XGB ssEPE model V4.pkl')
        feature_checkpoint = Path('model/Features.pkl')
        explainer_checkpoint = Path('model/explainer.pkl')
        shap_checkpoint = Path('model/model shap.pkl')

        # download from Google Drive if model or features are not present
        if not model_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(Model_location, model_checkpoint)
        if not feature_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(Feature_location, feature_checkpoint)

        model = joblib.load(model_checkpoint)
        features = joblib.load(feature_checkpoint)
        if not explainer_checkpoint.exists():
            explainer = shap.TreeExplainer(model, np.array(features), model_output='probability')
            joblib.dump(explainer, explainer_checkpoint)
        explainer2 = joblib.load(explainer_checkpoint)
        if not shap_checkpoint.exists():
            model_shap = explainer2(features)
            joblib.dump(model_shap, shap_checkpoint)
        model_shap2 = joblib.load(shap_checkpoint)
        return model, features, explainer2, model_shap2

    model, features, explainer, model_shap = load_items()

    # Load blank prostate as image objects from GitHub repository
    def load_images():
        image2 = PIL.Image.open('Images/Prostate diagram.png')
        return image2

    image2 = load_images()

    # Define choices and labels for feature inputs
    CHOICES = {0: 'No', 1: 'Yes'}

    def format_func_yn(option):
        return CHOICES[option]

    G_CHOICES = {0: 'Normal', 1: 'HGPIN', 2: 'ASAP', 3: 'Gleason 3+3', 4: 'Gleason 3+4', 5: 'Gleason 4+3',
                 6: 'Gleason 4+4',
                 7: 'Gleason 4+5/5+4'}

    def format_func_gleason(option):
        return G_CHOICES[option]

    features_list = ('PSA',
                     'Maximum % core involvement',
                     '% Gleason pattern 4/5',
                     'Perineural invasion',
                     'Base % core involvement',
                     'Base findings',
                     '% positive cores',
                     'Transition zone % core involvement',
                     'Age',
                     'Worst Gleason Grade Group',
                     'Mid % core involvement')

    # Input individual values in sidebar
    st.sidebar.header("Enter patient values")
    with st.sidebar:
        with st.form(key="my_form"):
            st.subheader("Global Variables")
            age = st.number_input("Age (years)", 0.0, 100.0, 60.0)
            psa = st.number_input("PSA (ng/ml)", 0.00, 200.00, 7.00)
            p_high = st.number_input("% Gleason pattern 4/5", 0.0, 100.00, 22.5)
            perineural_inv = st.selectbox("Perineural invasion", options=list(CHOICES.keys()),
                                          format_func=format_func_yn, index=1)

            st.subheader("Side-specific Variables - LEFT")
            base_findings = st.selectbox('Base findings', options=list(G_CHOICES.keys()),
                                         format_func=format_func_gleason, index=3)
            base_p_core = st.number_input('Base # of positive cores', 0, 10, value=1)
            base_t_core = st.number_input('Base # of cores taken', 0, 10, value=2)
            base_p_inv = st.number_input('Base % core involvement (0 to 100)', 0.0, 100.0, value=7.5)
            mid_findings = st.selectbox('Mid findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                        index=3)
            mid_p_core = st.number_input('Mid # of positive cores', 0, 10, value=1)
            mid_t_core = st.number_input('Mid # of cores taken', 0, 10, value=2)
            mid_p_inv = st.number_input('Mid % core involvement (0 to 100)', 0.0, 100.0, value=5.0)
            apex_findings = st.selectbox('Apex findings', options=list(G_CHOICES.keys()),
                                         format_func=format_func_gleason, index=0)
            apex_p_core = st.number_input('Apex # of positive cores', 0, 10, value=0)
            apex_t_core = st.number_input('Apex # of cores taken', 0, 10, value=1)
            apex_p_inv = st.number_input('Apex % core involvement (0 to 100)', 0.0, 100.0, value=0.0)
            tz_findings = st.selectbox('Transition zone findings', options=list(G_CHOICES.keys()),
                                       format_func=format_func_gleason, index=0)
            tz_p_core = st.number_input('Transition zone # of positive cores', 0, 10, value=0)
            tz_t_core = st.number_input('Transition zone # of cores taken', 0, 10, value=1)
            tz_p_inv = st.number_input('Transition zone % core involvement (0 to 100)', 0.0, 100.0, value=0.0)

            st.subheader("Side-specific Variables - RIGHT")
            base_findings_r = st.selectbox('Right Base findings', options=list(G_CHOICES.keys()),
                                           format_func=format_func_gleason, index=5)
            base_p_core_r = st.number_input('Right Base # of positive cores', 0, 10, value=2)
            base_t_core_r = st.number_input('Right Base # of cores taken', 0, 10, value=2)
            base_p_inv_r = st.number_input('Right Base % core involvement (0 to 100)', 0.0, 100.0, value=45.0)
            mid_findings_r = st.selectbox('Right Mid findings', options=list(G_CHOICES.keys()),
                                          format_func=format_func_gleason, index=4)
            mid_p_core_r = st.number_input('Right Mid # of positive cores', 0, 10, value=2)
            mid_t_core_r = st.number_input('Right Mid # of cores taken', 0, 10, value=2)
            mid_p_inv_r = st.number_input('Right Mid % core involvement (0 to 100)', 0.0, 100.0, value=45.0)
            apex_findings_r = st.selectbox('Right Apex findings', options=list(G_CHOICES.keys()),
                                           format_func=format_func_gleason, index=3)
            apex_p_core_r = st.number_input('Right Apex # of positive cores', 0, 10, value=1)
            apex_t_core_r = st.number_input('Right Apex # of cores taken', 0, 10, value=1)
            apex_p_inv_r = st.number_input('Right Apex % core involvement (0 to 100)', 0.0, 100.0, value=20.0)
            tz_findings_r = st.selectbox('Right Transition zone findings', options=list(G_CHOICES.keys()),
                                         format_func=format_func_gleason, index=4)
            tz_p_core_r = st.number_input('Right Transition zone # of positive cores', 0, 10, value=1)
            tz_t_core_r = st.number_input('Right Transition zone # of cores taken', 0, 10, value=1)
            tz_p_inv_r = st.number_input('Right Transition zone % core involvement (0 to 100)', 0.0, 100.0, value=80.0)

            submitted = st.form_submit_button(label='Submit')

            if submitted:
                ### LEFT DATA STORAGE ###
                # Group site findings into a list
                gleason_t = [base_findings, mid_findings, apex_findings, tz_findings]

                # Group % core involvements at each site into a list
                p_inv_t = [base_p_inv, mid_p_inv, apex_p_inv, tz_p_inv]

                # Combine site findings and % core involvements into a pandas DataFrame and sort by descending Gleason then
                # descending % core involvement
                g_p_inv = pd.DataFrame({'Gleason': gleason_t, '% core involvement': p_inv_t})
                sort_g_p_inv = g_p_inv.sort_values(by=['Gleason', '% core involvement'], ascending=False)

                # Calculate total positive cores and total cores taken for left lobe
                p_core_total = base_p_core + mid_p_core + apex_p_core + tz_p_core
                t_core_total = base_t_core + mid_t_core + apex_t_core + tz_t_core

                # Store a dictionary into a variable
                pt_data = {'PSA': psa,
                           'Maximum % core involvement': sort_g_p_inv['% core involvement'].max(),
                           '% Gleason pattern 4/5': p_high,
                           'Perineural invasion': perineural_inv,
                           'Base % core involvement': base_p_inv,
                           'Base findings': base_findings,
                           '% positive cores': round((p_core_total / t_core_total) * 100, 1),
                           'Transition zone % core involvement': tz_p_inv,
                           'Age': age,
                           'Worst Gleason Grade Group': sort_g_p_inv['Gleason'].max(),
                           'Mid % core involvement': mid_p_inv
                           }

                pt_features = pd.DataFrame(pt_data, index=[0])

                ### RIGHT DATA STORAGE ###
                # Group site findings into a list
                gleason_t_r = [base_findings_r, mid_findings_r, apex_findings_r, tz_findings_r]

                # Group % core involvements at each site into a list
                p_inv_t_r = [base_p_inv_r, mid_p_inv_r, apex_p_inv_r, tz_p_inv_r]

                # Combine site findings and % core involvements into a pandas DataFrame and sort by descending Gleason
                # then descending % core involvement
                g_p_inv_r = pd.DataFrame({'Gleason': gleason_t_r, '% core involvement': p_inv_t_r})
                sort_g_p_inv_r = g_p_inv_r.sort_values(by=['Gleason', '% core involvement'], ascending=False)

                # Calculate total positive cores and total cores taken for left lobe
                p_core_total_r = base_p_core_r + mid_p_core_r + apex_p_core_r + tz_p_core_r
                t_core_total_r = base_t_core_r + mid_t_core_r + apex_t_core_r + tz_t_core_r

                # Store a dictionary into a variable
                pt_data_r = {'PSA': psa,
                             'Maximum % core involvement': sort_g_p_inv_r['% core involvement'].max(),
                             '% Gleason pattern 4/5': p_high,
                             'Perineural invasion': perineural_inv,
                             'Base % core involvement': base_p_inv_r,
                             'Base findings': base_findings_r,
                             '% positive cores': round((p_core_total_r / t_core_total_r) * 100, 1),
                             'Transition zone % core involvement': tz_p_inv_r,
                             'Age': age,
                             'Worst Gleason Grade Group': sort_g_p_inv_r['Gleason'].max(),
                             'Mid % core involvement': mid_p_inv_r
                             }

                pt_features_r = pd.DataFrame(pt_data_r, index=[0])

                ### ANNOTATED PROSTATE DIAGRAM ###
                # Create text to overlay on annotated prostate diagram, auto-updates based on user inputted values
                base_L = str(G_CHOICES[base_findings]) + '\n' \
                         + 'Positive cores: ' + str(base_p_core) + '/' + str(base_t_core) + '\n' \
                         + '% core inv: ' + str(base_p_inv)
                mid_L = str(G_CHOICES[mid_findings]) + '\n' \
                        + 'Positive cores: ' + str(mid_p_core) + '/' + str(mid_t_core) + '\n' \
                        + '% core inv: ' + str(mid_p_inv)
                apex_L = str(G_CHOICES[apex_findings]) + '\n' \
                         + 'Positive cores: ' + str(apex_p_core) + '/' + str(apex_t_core) + '\n' \
                         + '% core inv: ' + str(apex_p_inv)
                tz_L = str(G_CHOICES[tz_findings]) + '\n' \
                       + 'Positive cores: ' + str(tz_p_core) + '/' + str(tz_t_core) + '\n' \
                       + '% core inv: ' + str(tz_p_inv)
                base_R = str(G_CHOICES[base_findings_r]) + '\n' \
                         + 'Positive cores: ' + str(base_p_core_r) + '/' + str(
                    base_t_core_r) + '\n' \
                         + '% core inv: ' + str(base_p_inv_r)
                mid_R = str(G_CHOICES[mid_findings_r]) + '\n' \
                        + 'Positive cores: ' + str(mid_p_core_r) + '/' + str(
                    mid_t_core_r) + '\n' \
                        + '% core inv: ' + str(mid_p_inv_r)
                apex_R = str(G_CHOICES[apex_findings_r]) + '\n' \
                         + 'Positive cores: ' + str(apex_p_core_r) + '/' + str(
                    apex_t_core_r) + '\n' \
                         + '% core inv: ' + str(apex_p_inv_r)
                tz_R = str(G_CHOICES[tz_findings_r]) + '\n' \
                       + 'Positive cores: ' + str(tz_p_core_r) + '/' + str(
                    tz_t_core_r) + '\n' \
                       + '% core inv: ' + str(tz_p_inv_r)

                # Set conditions to show colour coded site images based on Gleason Grade Group for each site
                draw = ImageDraw.Draw(image2)
                if base_findings == 3:
                    image_bl_G1 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason1.png')))
                    image2.paste(image_bl_G1, (145, 958), mask=image_bl_G1)
                if base_findings == 4:
                    image_bl_G2 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason2.png')))
                    image2.paste(image_bl_G2, (145, 958), mask=image_bl_G2)
                if base_findings == 5:
                    image_bl_G3 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason3.png')))
                    image2.paste(image_bl_G3, (145, 958), mask=image_bl_G3)
                if base_findings == 6:
                    image_bl_G4 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason4.png')))
                    image2.paste(image_bl_G4, (145, 958), mask=image_bl_G4)
                if base_findings == 7:
                    image_bl_G5 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason5.png')))
                    image2.paste(image_bl_G5, (145, 958), mask=image_bl_G5)

                if mid_findings == 3:
                    image_ml_G1 = PIL.Image.open('Images/Mid_Gleason1.png')
                    image2.paste(image_ml_G1, (145, 606), mask=image_ml_G1)
                if mid_findings == 4:
                    image_ml_G2 = PIL.Image.open('Images/Mid_Gleason2.png')
                    image2.paste(image_ml_G2, (145, 606), mask=image_ml_G2)
                if mid_findings == 5:
                    image_ml_G3 = PIL.Image.open('Images/Mid_Gleason3.png')
                    image2.paste(image_ml_G3, (145, 606), mask=image_ml_G3)
                if mid_findings == 6:
                    image_ml_G4 = PIL.Image.open('Images/Mid_Gleason4.png')
                    image2.paste(image_ml_G4, (145, 606), mask=image_ml_G4)
                if mid_findings == 7:
                    image_ml_G5 = PIL.Image.open('Images/Mid_Gleason5.png')
                    image2.paste(image_ml_G5, (145, 606), mask=image_ml_G5)

                if apex_findings == 3:
                    image_al_G1 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason1.png'))
                    image2.paste(image_al_G1, (145, 130), mask=image_al_G1)
                if apex_findings == 4:
                    image_al_G2 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason2.png'))
                    image2.paste(image_al_G2, (145, 130), mask=image_al_G2)
                if apex_findings == 5:
                    image_al_G3 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason3.png'))
                    image2.paste(image_al_G3, (145, 130), mask=image_al_G3)
                if apex_findings == 6:
                    image_al_G4 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason4.png'))
                    image2.paste(image_al_G4, (145, 130), mask=image_al_G4)
                if apex_findings == 7:
                    image_al_G5 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason5.png'))
                    image2.paste(image_al_G5, (145, 130), mask=image_al_G5)

                if tz_findings == 3:
                    image_tl_G1 = PIL.Image.open('Images/TZ_Gleason1.png')
                    image2.paste(image_tl_G1, (665, 493), mask=image_tl_G1)
                if tz_findings == 4:
                    image_tl_G2 = PIL.Image.open('Images/TZ_Gleason2.png')
                    image2.paste(image_tl_G2, (665, 493), mask=image_tl_G2)
                if tz_findings == 5:
                    image_tl_G3 = PIL.Image.open('Images/TZ_Gleason3.png')
                    image2.paste(image_tl_G3, (665, 493), mask=image_tl_G3)
                if tz_findings == 6:
                    image_tl_G4 = PIL.Image.open('Images/TZ_Gleason4.png')
                    image2.paste(image_tl_G4, (665, 493), mask=image_tl_G4)
                if tz_findings == 7:
                    image_tl_G5 = PIL.Image.open('Images/TZ_Gleason5.png')
                    image2.paste(image_tl_G5, (665, 493), mask=image_tl_G5)

                if base_findings_r == 3:
                    image_br_G1 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason1.png'))
                    image2.paste(image_br_G1, (1104, 958), mask=image_br_G1)
                if base_findings_r == 4:
                    image_br_G2 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason2.png'))
                    image2.paste(image_br_G2, (1104, 958), mask=image_br_G2)
                if base_findings_r == 5:
                    image_br_G3 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason3.png'))
                    image2.paste(image_br_G3, (1104, 958), mask=image_br_G3)
                if base_findings_r == 6:
                    image_br_G4 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason4.png'))
                    image2.paste(image_br_G4, (1104, 958), mask=image_br_G4)
                if base_findings_r == 7:
                    image_br_G5 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason5.png'))
                    image2.paste(image_br_G5, (1104, 958), mask=image_br_G5)

                if mid_findings_r == 3:
                    image_mr_G1 = PIL.Image.open('Images/Mid_Gleason1.png')
                    image2.paste(image_mr_G1, (1542, 606), mask=image_mr_G1)
                if mid_findings_r == 4:
                    image_mr_G2 = PIL.Image.open('Images/Mid_Gleason2.png')
                    image2.paste(image_mr_G2, (1542, 606), mask=image_mr_G2)
                if mid_findings_r == 5:
                    image_mr_G3 = PIL.Image.open('Images/Mid_Gleason3.png')
                    image2.paste(image_mr_G3, (1542, 606), mask=image_mr_G3)
                if mid_findings_r == 6:
                    image_mr_G4 = PIL.Image.open('Images/Mid_Gleason4.png')
                    image2.paste(image_mr_G4, (1542, 606), mask=image_mr_G4)
                if mid_findings_r == 7:
                    image_mr_G5 = PIL.Image.open('Images/Mid_Gleason5.png')
                    image2.paste(image_mr_G5, (1542, 606), mask=image_mr_G5)

                if apex_findings_r == 3:
                    image_ar_G1 = PIL.Image.open('Images/Corner_Gleason1.png')
                    image2.paste(image_ar_G1, (1104, 130), mask=image_ar_G1)
                if apex_findings_r == 4:
                    image_ar_G2 = PIL.Image.open('Images/Corner_Gleason2.png')
                    image2.paste(image_ar_G2, (1104, 130), mask=image_ar_G2)
                if apex_findings_r == 5:
                    image_ar_G3 = PIL.Image.open('Images/Corner_Gleason3.png')
                    image2.paste(image_ar_G3, (1104, 130), mask=image_ar_G3)
                if apex_findings_r == 6:
                    image_ar_G4 = PIL.Image.open('Images/Corner_Gleason4.png')
                    image2.paste(image_ar_G4, (1104, 130), mask=image_ar_G4)
                if apex_findings_r == 7:
                    image_ar_G5 = PIL.Image.open('Images/Corner_Gleason5.png')
                    image2.paste(image_ar_G5, (1104, 130), mask=image_ar_G5)

                if tz_findings_r == 3:
                    image_tr_G1 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason1.png'))
                    image2.paste(image_tr_G1, (1100, 493), mask=image_tr_G1)
                if tz_findings_r == 4:
                    image_tr_G2 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason2.png'))
                    image2.paste(image_tr_G2, (1100, 493), mask=image_tr_G2)
                if tz_findings_r == 5:
                    image_tr_G3 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason3.png'))
                    image2.paste(image_tr_G3, (1100, 493), mask=image_tr_G3)
                if tz_findings_r == 6:
                    image_tr_G4 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason4.png'))
                    image2.paste(image_tr_G4, (1100, 493), mask=image_tr_G4)
                if tz_findings_r == 7:
                    image_tr_G5 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason5.png'))
                    image2.paste(image_tr_G5, (1100, 493), mask=image_tr_G5)

                # Overlay text showing Gleason Grade Group, % positive cores, and % core involvement for each site
                draw.text((525, 1110), base_L, fill="black", font=font, align="center")
                draw.text((205, 690), mid_L, fill="black", font=font, align="center")
                draw.text((525, 275), apex_L, fill="black", font=font, align="center")
                draw.text((685, 690), tz_L, fill="black", font=font, align="center")
                draw.text((1300, 1110), base_R, fill="black", font=font, align="center")
                draw.text((1590, 690), mid_R, fill="black", font=font, align="center")
                draw.text((1300, 275), apex_R, fill="black", font=font, align="center")
                draw.text((1125, 690), tz_R, fill="black", font=font, align="center")
                col1.image(image2, use_column_width='auto')
                col1.write('**Red bars**: Features that ***increase*** the risk of ssEPE  \n'
                           '**Blue bars**: Features that ***decrease*** the risk of ssEPE  \n'
                           '**Width of bars**: Importance of the feature. The wider it is, the greater impact it has '
                           'on risk of ssEPE')

                ### SHAP FORCE PLOTS ###
                # SHAP plot for left lobe
                col2.subheader('Left lobe')
                st.set_option('deprecation.showPyplotGlobalUse', False)

                shap_values = explainer.shap_values(pt_features)
                shap.force_plot(0.3, shap_values, pt_features, features_list, text_rotation=10,  # features_list,
                                matplotlib=True)
                col2.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')
                plt.clf()

                # SHAP plot for right lobe
                col2.subheader('Right lobe')
                shap_values_r = explainer.shap_values(pt_features_r)
                shap.plots.force(0.3, shap_values_r, pt_features_r, features_list, matplotlib=True,
                                 text_rotation=10)
                col2.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')
                plt.clf()

                ### COMPARISON TO STUDY POPULATION ###
                colglobal.write('**Global Variables**')

                # PSA
                shap.plots.scatter(model_shap[:, 0], hist=True, dot_size=5, show=False)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_psa = np.array(pt_features)[:, 0]
                y_pt_psa = shap_values[:, 0]
                plt.plot(x_pt_psa, y_pt_psa, 'ro', markersize=7, alpha=1)
                colpsa.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # Maximum % core involvement
                shap.plots.scatter(model_shap[:, 1], hist=True, dot_size=5, show=False)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_maxci = np.array(pt_features)[:, 1]
                y_pt_maxci = shap_values[:, 1]
                plt.plot(x_pt_maxci, y_pt_maxci, 'ro', markersize=7, alpha=1)
                colmaxci.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # % Gleason 4/5
                shap.plots.scatter(model_shap[:, 2], hist=True, dot_size=5, show=False)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_phigh = np.array(pt_features)[:, 2]
                y_pt_phigh = shap_values[:, 2]
                plt.plot(x_pt_phigh, y_pt_phigh, 'ro', markersize=7, alpha=1)
                colphigh.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # Perineural invasion
                shap.plots.scatter(model_shap[:, 3], hist=True, dot_size=5, show=False)
                positions = (0, 1)
                x_labels = ('No', 'Yes')
                plt.xticks(positions, x_labels, rotation=0)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_pinv = np.array(pt_features)[:, 3]
                y_pt_pinv = shap_values[:, 3]
                plt.plot(x_pt_pinv, y_pt_pinv, 'ro', markersize=7, alpha=1)
                colpinv.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # Age
                shap.plots.scatter(model_shap[:, 8], hist=True, dot_size=5, show=False)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_age = np.array(pt_features)[:, 8]
                y_pt_age = shap_values[:, 8]
                plt.plot(x_pt_age, y_pt_age, 'ro', markersize=7, alpha=1)
                colage.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                colside.write('**Side-specific Variables**')
                colside.write('**Red** data-points represents **left** side-specific values, while **green** '
                              'data-points represents **right** side-specific values.')

                # Base # core involvement
                shap.plots.scatter(model_shap[:, 4], hist=True, dot_size=5, show=False)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_bci = np.array(pt_features)[:, 4]
                y_pt_bci = shap_values[:, 4]
                x_pt_bcir = np.array(pt_features_r)[:, 4]
                y_pt_bcir = shap_values_r[:, 4]
                plt.plot(x_pt_bci, y_pt_bci, 'ro', markersize=7, alpha=1, color='red')
                plt.plot(x_pt_bcir, y_pt_bcir, 'ro', markersize=7, alpha=1, color='green')
                colbci.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # Mid % core involvement
                shap.plots.scatter(model_shap[:, 10], hist=True, dot_size=5, show=False)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_mci = np.array(pt_features)[:, 10]
                y_pt_mci = shap_values[:, 10]
                x_pt_mcir = np.array(pt_features_r)[:, 10]
                y_pt_mcir = shap_values_r[:, 10]
                plt.plot(x_pt_mci, y_pt_mci, 'ro', markersize=7, alpha=1, color='red')
                plt.plot(x_pt_mcir, y_pt_mcir, 'ro', markersize=7, alpha=1, color='green')
                colmci.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # TZ % core involvement
                shap.plots.scatter(model_shap[:, 7], hist=True, dot_size=5, show=False)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_tzci = np.array(pt_features)[:, 7]
                y_pt_tzci = shap_values[:, 7]
                x_pt_tzcir = np.array(pt_features_r)[:, 7]
                y_pt_tzcir = shap_values_r[:, 7]
                plt.plot(x_pt_tzci, y_pt_tzci, 'ro', markersize=7, alpha=1, color='red')
                plt.plot(x_pt_tzcir, y_pt_tzcir, 'ro', markersize=7, alpha=1, color='green')
                coltzci.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # Base findings
                shap.plots.scatter(model_shap[:, 5], hist=True, dot_size=5, show=False)
                positions = (0, 1, 2, 3, 4, 5, 6, 7)
                x_labels = ('Normal', 'HGPIN', 'ASAP', 'GGG1', 'GGG2', 'GGG3', 'GGG4', 'GGG5')
                plt.xticks(positions, x_labels, rotation=0)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_bf = np.array(pt_features)[:, 5]
                y_pt_bf = shap_values[:, 5]
                x_pt_bfr = np.array(pt_features_r)[:, 5]
                y_pt_bfr = shap_values_r[:, 5]
                plt.plot(x_pt_bf, y_pt_bf, 'ro', markersize=7, alpha=1, color='red')
                plt.plot(x_pt_bfr, y_pt_bfr, 'ro', markersize=7, alpha=1, color='green')
                colbf.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # Worst Gleason Grade Group
                shap.plots.scatter(model_shap[:, 9], hist=True, dot_size=5, show=False)
                positions = (0, 1, 2, 3, 4, 5, 6, 7)
                x_labels = ('Normal', 'HGPIN', 'ASAP', 'GGG1', 'GGG2', 'GGG3', 'GGG4', 'GGG5')
                plt.xticks(positions, x_labels, rotation=0)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_wggg = np.array(pt_features)[:, 9]
                y_pt_wggg = shap_values[:, 9]
                x_pt_wgggr = np.array(pt_features_r)[:, 9]
                y_pt_wgggr = shap_values_r[:, 9]
                plt.plot(x_pt_wggg, y_pt_wggg, 'ro', markersize=7, alpha=1, color='red')
                plt.plot(x_pt_wgggr, y_pt_wgggr, 'ro', markersize=7, alpha=1, color='green')
                colwggg.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')

                # % positive cores
                shap.plots.scatter(model_shap[:, 6], hist=True, dot_size=5, show=False)
                plt.ylabel('Impact on probability of ssEPE')
                x_pt_pc = np.array(pt_features)[:, 6]
                y_pt_pc = shap_values[:, 6]
                x_pt_pcr = np.array(pt_features_r)[:, 6]
                y_pt_pcr = shap_values_r[:, 6]
                plt.plot(x_pt_pc, y_pt_pc, 'ro', markersize=7, alpha=1, color='red')
                plt.plot(x_pt_pcr, y_pt_pcr, 'ro', markersize=7, alpha=1, color='green')
                colpc.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')            

def about(session_state):
    st.markdown(
        """
    Welcome to the Side-Specific Extraprostatic Extension (ssEPE) prediction tool. ssEPE was developed to provide 
    three specific outputs that may be beneficial for surgical planning and patient counselling:
    * Annotated prostate diagram showing location and severity of disease based on prostate biopsy
    * Probability of ssEPE for the left and right prostatic lobe
    * Comparison of individual patient characteristics to the study population used to create the ssEPE tool

    The ssEPE tool achieved an area under the receiver-operating-characteristic (AUROC) of 0.81 for both the training 
    cohort (stratified 10-fold cross-validation) and external testing cohort. The ssEPE tool outperformed existing 
    biopsy-derived prediction models. Additional information can be found in the reference below or the Model 
    Development tab. 

    """
    )
    st.subheader("Reference")
    st.markdown(
        """
    **Explainable artificial intelligence to predict the risk of side-specific extraprostatic extension in 
    pre-prostatectomy patients**\n
    
    *Jethro CC. Kwong$^{1,2}$, Adree Khondker$^{3}$, Christopher Tran$^{3}$, Emily Evans$^{3}$, Amna Ali$^{4}$, 
    Munir Jamal$^{1}$, Thomas Short$^{1}$, Frank Papanikolaou$^{1}$, John R. Srigley$^{5}$, Andrew H. Feifer$^{1,4}$*\n
    
    1. Division of Urology, Department of Surgery, University of Toronto, Toronto, ON, Canada
    1. Temerty Centre for AI Research and Education in Medicine, University of Toronto, Toronto, Canada
    1. Temerty Faculty of Medicine, University of Toronto, Toronto, ON, Canada
    1. Institute for Better Health, Trillium Health Partners, Mississauga, ON, Canada
    1. Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, ON, Canada
  
    """
    )

def dev(session_state):
    @st.cache(allow_output_mutation=True)
    def load_static_images():
        auroc_train = PIL.Image.open('Performance Metrics/AUROC train.png')
        auroc_test = PIL.Image.open('Performance Metrics/AUROC test.png')
        auprc_train = PIL.Image.open('Performance Metrics/AUPRC train.png')
        auprc_test = PIL.Image.open('Performance Metrics/AUPRC test.png')
        calib_train = PIL.Image.open('Performance Metrics/Calibration train.png')
        calib_test = PIL.Image.open('Performance Metrics/Calibration test.png')
        dca = PIL.Image.open('Performance Metrics/DCA.png')
        stream_uro = PIL.Image.open('Performance Metrics/ssEPE STREAM-URO.png')
        summary = PIL.Image.open('Performance Metrics/Feature rankings.png')
        pdp = PIL.Image.open('Performance Metrics/Partial dependence plots.png')
        return auroc_train, auroc_test, auprc_train, auprc_test, calib_train, calib_test, \
               dca, stream_uro, summary, pdp

    auroc_train, auroc_test, auprc_train, auprc_test, calib_train, calib_test, \
    dca, stream_uro, summary, pdp = load_static_images()

    st.header("How the model was developed")
    st.write("""""")
    st.markdown(
        """
    A retrospective sample of 900 prostatic lobes (450 patients) from radical prostatectomy (RP) specimens at
    Credit Valley Hospital, Mississauga, between 2010 and 2020, was used as the training cohort. Features
    (ie: variables) included patient demographics, clinical, and site-specific data from
    transrectal ultrasound-guided prostate biopsy. The primary label (ie: outcome) of interest was the presence
    of EPE in the ipsilateral lobe of the prostatectomy specimen. A previously developed [model]
    (https://bjui-journals.onlinelibrary.wiley.com/doi/full/10.1111/bju.13733), which has the highest performance out of
    current biopsy-derived predictive models for ssEPE that have been externally validated,
    was used as the baseline model for comparison. We also developed a separate logistic regression (LR) model using the
    same features included in our machine learning model for comparison. \n
    
    Dimensionality reduction was performed using a modified [Boruta](https://www.jstatsoft.org/article/view/v036i11/0)
    algorithm followed by removing highly correlated features (Pearson correlation > 0.8). This former involves
    fitting all features to a random forest model and determining feature importance
    by comparing the relevance of each feature to that of random noise. Given that our dataset contains both
    categorical and numerical features, SHAP was specifically selected in lieu of impurity-based measures
    to reduce bias towards high cardinality features. \n
    
    Using the final set of the most important and independent features, a stratified ten-fold 
    cross-validation method was performed to train a gradient-boosted machine, optimize hyperparameters,
    and for internal validation. In stratified cross-validation, the training cohort was randomly partitioned
    into ten equal folds, with each fold containing the same percentage of positive ssEPE cases. Nine folds
    were used for model training and hyperparameter tuning while the remaining fold made up the validation cohort.
    This process was repeated ten times such that each fold served as the validation cohort once. Model
    performance was determined based on the average performance across all ten validation cohorts to improve
    generalizability of the models. All models were further externally validated using a testing cohort of
    122 lobes (61 patients) from RP specimens at Mississauga Hospital, Mississauga, between 2016 and 2020.
    All models were assessed by area under receiver-operating-characteristic curve (AUROC),
    precision-recall curve (AUPRC), calibration curve, and decision curve analysis. The incidence of ssEPE in
    the training and testing cohorts were 30.7 and 41.8%, respectively.
    """
    )
    st.write("""""")
    st.write("""""")
    colROC, colPRC = st.columns([1, 1])
    colROC.write('**Area under receiver-operating-characteristic curve (AUROC):** is used to measure the discriminative\
                   capability of predictive models by comparing the true positive rate (sensitivity) and false positive\
                   rate (1-specificity) across various decision thresholds.')
    colROC.write('Our ML model achieved the highest AUROC with a **mean AUROC of 0.81** (95% CI 0.78-0.83) followed by\
                    LR 0.78 (95% CI 0.75-0.80, p<0.01) and baseline 0.74 (95% CI 0.71-0.76, p<0.01) on cross-validation of\
                     the training cohort. Similarly, our ML model performed favourably on the external testing cohort with\
                      an **AUROC of 0.81** (95% CI 0.73-0.88) compared to LR 0.76 (95% CI 0.67-0.83, p=0.01) and baseline \
                    0.75 (95% CI 0.67-0.83, p=0.03).')
    colPRC.write('**Area under precision-recall curve (AUPRC):** compares precision (positive predictive value) and \
                       recall (sensitivity) across various decision thresholds. It is more informative than AUROC curves when\
                       evaluating the performance of classifiers for imbalanced datasets, such as in our case where there are\
                        more patients without ssEPE than with ssEPE. This is because AUPRC evaluates the proportion of true\
                       positives among positive predictions, which is our outcome of interest.')
    colPRC.write('Our ML model achieved the highest AUPRC with a **mean AUPRC of 0.69** (95% CI 0.64-0.73) followed by \
                        LR 0.64 (95% CI 0.59-0.69) and baseline 0.59 (95% CI 0.54-0.65) on cross-validation of the training \
                        cohort. Similarly, our ML model performed favourably on the external testing cohort with an \
                        **AUPRC of 0.78** (95% CI 0.67-0.86) compared to LR 0.75 (95% CI 0.65-0.84) and baseline 0.70 (95% CI \
                        0.60-0.79).')
    colF, colG, colH, colI = st.columns([1, 1, 1, 1])
    colF.image(auroc_train, use_column_width='auto')
    colG.image(auroc_test, use_column_width='auto')
    colH.image(auprc_train, use_column_width='auto')
    colI.image(auprc_test, use_column_width='auto')
    st.write("""""")
    st.write("""""")
    colCal, colDCA = st.columns([1, 1])
    colCal.write('**Calibration curves:** are used to evaluate the accuracy of model risk estimates by measuring the\
                   agreement between the predicted and observed number of outcomes. A perfectly calibrated model is\
                   depicted as a 45 degree line. In our case, if a calibration curve is above the reference line, it\
                    underestimates the risk of ssEPE, which may lead to undertreatment (ie: leaving some cancer behind).\
                     However, if a calibration curve is below the reference line, it overestimates the risk of ssEPE, which\
                      may lead to overtreatment (ie: patient gets unnecessarily treated with a non-nerve sparing approach).\
                       Therefore, calibration is especially important when evaluating predictive models used to support\
                        decision-making.')
    colCal.write('Our ML model is well calibrated for predicted probabilities between 0-40%, while overestimating the risk\
                    of ssEPE above 40% probability in the testing cohort.')
    colDCA.write('**[Decision curve analysis](https://pubmed.ncbi.nlm.nih.gov/17099194/):** is used to evaluate clinical\
                    utility. Here, the net benefit of the model is plotted against various threshold probabilities for\
                     three different treatment strategies: treat all, treat none, or treat only those predicted to have\
                      ssEPE by the model.')
    colDCA.write('Threshold probabilities between 10-30% were deemed the most clinically relevant for consideration\
                     of nerve-sparing. Our ML model achieved the highest net benefit across these\
                     thresholds. This translates to a potential **increase in appropriate nerve-sparing by 14 (ML) vs\
                     8 (LR) vs 1 (baseline) per 100 cases at a threshold probability of 15%** compared to a "treat all"\
                      strategy.')
    colJ, colK, colL = st.columns([1, 1, 2])
    colJ.image(calib_train, use_column_width='auto')
    colK.image(calib_test, use_column_width='auto')
    colL.image(dca, use_column_width='auto')

    st.write("""""")
    st.write("""""")
    st.write('This model was developed in accordance to the STREAM-URO framework (see table below).')
    st.image(stream_uro, width=900)
    st.write("""""")
    st.write("""""")

    st.header("How the model explanations were determined")
    st.write("""""")
    st.markdown(
        """
    Model explanations were calculated based on SHAP (SHapley Additive exPlanations) values,
    originally developed by [Lundberg and Lee (2017)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf).\
    This is an additive feature attribution method that satisfies all three properties of explanation models: local accuracy, missingness, and consistency.')
    **Accuracy**: the output of the explanation model must match the output of the original model for a\
    given prediction.  \n
    **Missingness**: when a feature is missing, it should have no impact on the model.  \n
    **Consistency**: a feature’s assigned attribution must match its degree of importance in the original model\
    (ie: if overall – % tissue involvement has the highest attribution, it must also have the highest feature importance\
    and the model must rely on this feature the most for a given prediction). \n
    
    SHAP allows us to understand why our model made a given prediction by simplifying our complex model into a\
    linear function of binary variables. This approach has previously been implemented to improve understanding\
    of [hypoxemia risk during anesthetic care](https://www.nature.com/articles/s41551-018-0304-0).
    """
    )
    st.write("""""")
    st.write("""""")

    st.header("Additional model explanations")
    st.write("""""")
    colA, colB = st.columns([1, 2])
    colA.write("**Feature importance rankings:** helps identify which features had the overall greatest impact on\
                 our ML model's predictions. Here, we see that PSA, Maximum % core involvement, and perineural invasion\
                 were the three most important features in our ML model.")
    colA.image(summary, use_column_width='auto')
    colB.write('**Partial dependence plots:** allows us to visualize how a given feature can impact the probability of \
                 ssEPE across all its possible values (ie: how does % Gleason pattern 4/5, from 0 to 100%, positively or\
                  negatively impact probability of ssEPE?). We see that our ML model represents each feature in different\
                 ways. Some have a linear or logarithmic relationship, while others are more complex.')
    colB.image(pdp, use_column_width='auto')


if __name__ == "__main__":
    st.set_page_config(page_title="ssEPE - Side-Specific Extraprostatic Extension Tool",
                       page_icon=":toilet:",
                       layout="wide",
                       initial_sidebar_state="expanded"
                       )
    main()

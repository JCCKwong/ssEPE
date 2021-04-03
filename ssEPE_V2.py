"""
Title: Explainable artificial intelligence to predict the risk of side-specific extraprostatic extension in
pre-prostatectomy patients

Developed by: Jethro CC. Kwong (1,2), Adree Khondker (3), Christopher Tran (3), Emily Evans (3), Amna Ali (4),
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

This model was developed in accordance to the STREAM-URO framework (to-be published)
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


# Default widescreen mode
def _max_width_():
    max_width_str = f"max-width: 1920px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

# Create a title for web app
st.title('Side-specific extraprostatic extension (ssEPE) prediction')
st.write('Determine the probability of ssEPE in the ipsilateral lobe using clinicopathological features and explainable\
 machine learning')

# Load saved items from Google Drive
GD_model_location = '1RrEJWueJLibz_LBdpR-Qu3lsAnZxN1tV'
GD_feature_location = '1ih8IQ4pSfGDS-slNWwQ-Ojn3JO4h54lP'


@st.cache(allow_output_mutation=True)
def load_items():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    f_checkpoint = Path('model/XGB ssEPE model V4.pkl')
    f_checkpoint1 = Path('model/Features.pkl')
    f_checkpoint2 = Path('model/explainer.pkl')

    # download from Google Drive if model or features are not present
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdd.download_file_from_google_drive(GD_model_location, f_checkpoint)
    if not f_checkpoint1.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdd.download_file_from_google_drive(GD_feature_location, f_checkpoint1)

    model = joblib.load(f_checkpoint)
    features = joblib.load(f_checkpoint1)
    if not f_checkpoint2.exists():
        explainer = shap.TreeExplainer(model, np.array(features), model_output='probability')
        joblib.dump(explainer, f_checkpoint2)
    explainer2 = joblib.load(f_checkpoint2)
    return model, explainer2


model, explainer = load_items()


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
    return auroc_train, auroc_test, auprc_train, auprc_test, calib_train, calib_test,\
           dca, stream_uro, summary, pdp


auroc_train, auroc_test, auprc_train, auprc_test, calib_train, calib_test,\
dca, stream_uro, summary, pdp = load_static_images()


# Load blank prostate and all colour coded sites as image objects from GitHub repository
def load_images():
    image2 = PIL.Image.open('Images/Prostate diagram.png')
    image_bl_G1 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason1.png')))
    image_bl_G2 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason2.png')))
    image_bl_G3 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason3.png')))
    image_bl_G4 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason4.png')))
    image_bl_G5 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason5.png')))
    image_ml_G1 = PIL.Image.open('Images/Mid_Gleason1.png')
    image_ml_G2 = PIL.Image.open('Images/Mid_Gleason2.png')
    image_ml_G3 = PIL.Image.open('Images/Mid_Gleason3.png')
    image_ml_G4 = PIL.Image.open('Images/Mid_Gleason4.png')
    image_ml_G5 = PIL.Image.open('Images/Mid_Gleason5.png')
    image_al_G1 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason1.png'))
    image_al_G2 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason2.png'))
    image_al_G3 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason3.png'))
    image_al_G4 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason4.png'))
    image_al_G5 = PIL.ImageOps.mirror(PIL.Image.open('Images/Corner_Gleason5.png'))
    image_tl_G1 = PIL.Image.open('Images/TZ_Gleason1.png')
    image_tl_G2 = PIL.Image.open('Images/TZ_Gleason2.png')
    image_tl_G3 = PIL.Image.open('Images/TZ_Gleason3.png')
    image_tl_G4 = PIL.Image.open('Images/TZ_Gleason4.png')
    image_tl_G5 = PIL.Image.open('Images/TZ_Gleason5.png')
    image_br_G1 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason1.png'))
    image_br_G2 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason2.png'))
    image_br_G3 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason3.png'))
    image_br_G4 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason4.png'))
    image_br_G5 = PIL.ImageOps.flip(PIL.Image.open('Images/Corner_Gleason5.png'))
    image_mr_G1 = PIL.Image.open('Images/Mid_Gleason1.png')
    image_mr_G2 = PIL.Image.open('Images/Mid_Gleason2.png')
    image_mr_G3 = PIL.Image.open('Images/Mid_Gleason3.png')
    image_mr_G4 = PIL.Image.open('Images/Mid_Gleason4.png')
    image_mr_G5 = PIL.Image.open('Images/Mid_Gleason5.png')
    image_ar_G1 = PIL.Image.open('Images/Corner_Gleason1.png')
    image_ar_G2 = PIL.Image.open('Images/Corner_Gleason2.png')
    image_ar_G3 = PIL.Image.open('Images/Corner_Gleason3.png')
    image_ar_G4 = PIL.Image.open('Images/Corner_Gleason4.png')
    image_ar_G5 = PIL.Image.open('Images/Corner_Gleason5.png')
    image_tr_G1 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason1.png'))
    image_tr_G2 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason2.png'))
    image_tr_G3 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason3.png'))
    image_tr_G4 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason4.png'))
    image_tr_G5 = PIL.ImageOps.mirror(PIL.Image.open('Images/TZ_Gleason5.png'))
    return image2, image_bl_G1, image_bl_G2, image_bl_G3, image_bl_G4, image_bl_G5, \
           image_ml_G1, image_ml_G2, image_ml_G3, image_ml_G4, image_ml_G5, \
           image_al_G1, image_al_G2, image_al_G3, image_al_G4, image_al_G5, \
           image_tl_G1, image_tl_G2, image_tl_G3, image_tl_G4, image_tl_G5, \
           image_br_G1, image_br_G2, image_br_G3, image_br_G4, image_br_G5, \
           image_mr_G1, image_mr_G2, image_mr_G3, image_mr_G4, image_mr_G5, \
           image_ar_G1, image_ar_G2, image_ar_G3, image_ar_G4, image_ar_G5, \
           image_tr_G1, image_tr_G2, image_tr_G3, image_tr_G4, image_tr_G5


image2, image_bl_G1, image_bl_G2, image_bl_G3, image_bl_G4, image_bl_G5, \
image_ml_G1, image_ml_G2, image_ml_G3, image_ml_G4, image_ml_G5, \
image_al_G1, image_al_G2, image_al_G3, image_al_G4, image_al_G5, \
image_tl_G1, image_tl_G2, image_tl_G3, image_tl_G4, image_tl_G5, \
image_br_G1, image_br_G2, image_br_G3, image_br_G4, image_br_G5, \
image_mr_G1, image_mr_G2, image_mr_G3, image_mr_G4, image_mr_G5, \
image_ar_G1, image_ar_G2, image_ar_G3, image_ar_G4, image_ar_G5, \
image_tr_G1, image_tr_G2, image_tr_G3, image_tr_G4, image_tr_G5 = load_images()

# Define choices and labels for feature inputs
CHOICES = {0: 'No', 1: 'Yes'}


def format_func_yn(option):
    return CHOICES[option]


G_CHOICES = {0: 'Normal', 1: 'HGPIN', 2: 'ASAP', 3: 'Gleason 3+3', 4: 'Gleason 3+4', 5: 'Gleason 4+3', 6: 'Gleason 4+4',
             7: 'Gleason 4+5/5+4'}


def format_func_gleason(option):
    return G_CHOICES[option]


# Create sidebar for user inputted values
st.sidebar.write('Enter patient values')


# Create sidebar inputs for global variables and left lobe variables
def get_user_input():
    with st.sidebar.beta_expander('Global variables', expanded=True):
        age = st.number_input('Age (years)', 0.0, 100.0, 60.0)
        psa = st.number_input('PSA (ng/ml)', 0.00, 200.00, 7.00)
        p_high = st.number_input('% Gleason pattern 4/5', 0.0, 100.00, 22.5)
        perineural_inv = st.selectbox('Perineural invasion', options=list(CHOICES.keys()), format_func=format_func_yn,
                                      index=1)
    with st.sidebar.beta_expander('Side-specific variables (Left)', expanded=True):
        base_findings = st.selectbox('Base findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                     key=0, index=3)
        base_p_core = st.number_input('Base # of positive cores', 0, 10, value=1, key=0)
        base_t_core = st.number_input('Base # of cores taken', 0, 10, value=2, key=0)
        base_p_inv = st.number_input('Base % core involvement (0 to 100)', 0.0, 100.0, value=7.5, key=0)
        mid_findings = st.selectbox('Mid findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                    key=0, index=3)
        mid_p_core = st.number_input('Mid # of positive cores', 0, 10, value=1, key=0)
        mid_t_core = st.number_input('Mid # of cores taken', 0, 10, value=2, key=0)
        mid_p_inv = st.number_input('Mid % core involvement (0 to 100)', 0.0, 100.0, value=5.0, key=0)
        apex_findings = st.selectbox('Apex findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                     key=0, index=0)
        apex_p_core = st.number_input('Apex # of positive cores', 0, 10, value=0, key=0)
        apex_t_core = st.number_input('Apex # of cores taken', 0, 10, value=1, key=0)
        apex_p_inv = st.number_input('Apex % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=0)
        tz_findings = st.selectbox('Transition zone findings', options=list(G_CHOICES.keys()),
                                   format_func=format_func_gleason, key=0, index=0)
        tz_p_core = st.number_input('Transition zone # of positive cores', 0, 10, value=0, key=0)
        tz_t_core = st.number_input('Transition zone # of cores taken', 0, 10, value=1, key=0)
        tz_p_inv = st.number_input('Transition zone % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=0)

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

    # Save positive cores and cores taken at each site for annotated prostate diagram later
    get_user_input.b_findings = base_findings
    get_user_input.b_p_core = base_p_core
    get_user_input.b_t_core = base_t_core
    get_user_input.b_p = base_p_inv
    get_user_input.m_findings = mid_findings
    get_user_input.m_p_core = mid_p_core
    get_user_input.m_t_core = mid_t_core
    get_user_input.m_p = mid_p_inv
    get_user_input.a_findings = apex_findings
    get_user_input.a_p_core = apex_p_core
    get_user_input.a_t_core = apex_t_core
    get_user_input.a_p = apex_p_inv
    get_user_input.t_findings = tz_findings
    get_user_input.t_p_core = tz_p_core
    get_user_input.t_t_core = tz_t_core
    get_user_input.t_p = tz_p_inv

    pt_features = pd.DataFrame(pt_data, index=[0])
    return pt_features


# Store the left lobe user input into a variable
user_input = get_user_input()


def get_user_input_r():
    with st.sidebar.beta_expander('Side-specific variables (Right)', expanded=True):
        base_findings_r = st.selectbox('Base findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                       key=1, index=5)
        base_p_core_r = st.number_input('Base # of positive cores', 0, 10, value=2, key=1)
        base_t_core_r = st.number_input('Base # of cores taken', 0, 10, value=2, key=1)
        base_p_inv_r = st.number_input('Base % core involvement (0 to 100)', 0.0, 100.0, value=45.0, key=1)
        mid_findings_r = st.selectbox('Mid findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                      key=1, index=4)
        mid_p_core_r = st.number_input('Mid # of positive cores', 0, 10, value=2, key=1)
        mid_t_core_r = st.number_input('Mid # of cores taken', 0, 10, value=2, key=1)
        mid_p_inv_r = st.number_input('Mid % core involvement (0 to 100)', 0.0, 100.0, value=45.0, key=1)
        apex_findings_r = st.selectbox('Apex findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                       key=1, index=3)
        apex_p_core_r = st.number_input('Apex # of positive cores', 0, 10, value=1, key=1)
        apex_t_core_r = st.number_input('Apex # of cores taken', 0, 10, value=1, key=1)
        apex_p_inv_r = st.number_input('Apex % core involvement (0 to 100)', 0.0, 100.0, value=20.0, key=1)
        tz_findings_r = st.selectbox('Transition zone findings', options=list(G_CHOICES.keys()),
                                     format_func=format_func_gleason, key=1, index=4)
        tz_p_core_r = st.number_input('Transition zone # of positive cores', 0, 10, value=1, key=1)
        tz_t_core_r = st.number_input('Transition zone # of cores taken', 0, 10, value=1, key=1)
        tz_p_inv_r = st.number_input('Transition zone % core involvement (0 to 100)', 0.0, 100.0, value=80.0, key=1)

    # Group site findings into a list
    gleason_t_r = [base_findings_r, mid_findings_r, apex_findings_r, tz_findings_r]

    # Group % core involvements at each site into a list
    p_inv_t_r = [base_p_inv_r, mid_p_inv_r, apex_p_inv_r, tz_p_inv_r]

    # Combine site findings and % core involvements into a pandas DataFrame and sort by descending Gleason then
    # descending % core involvement
    g_p_inv_r = pd.DataFrame({'Gleason': gleason_t_r, '% core involvement': p_inv_t_r})
    sort_g_p_inv_r = g_p_inv_r.sort_values(by=['Gleason', '% core involvement'], ascending=False)

    # Calculate total positive cores and total cores taken for left lobe
    p_core_total_r = base_p_core_r + mid_p_core_r + apex_p_core_r + tz_p_core_r
    t_core_total_r = base_t_core_r + mid_t_core_r + apex_t_core_r + tz_t_core_r

    # Store a dictionary into a variable
    pt_data_r = {'PSA': user_input['PSA'],
                 'Maximum % core involvement': sort_g_p_inv_r['% core involvement'].max(),
                 '% Gleason pattern 4/5': user_input['% Gleason pattern 4/5'],
                 'Perineural invasion': user_input['Perineural invasion'],
                 'Base % core involvement': base_p_inv_r,
                 'Base findings': base_findings_r,
                 '% positive cores': round((p_core_total_r / t_core_total_r) * 100, 1),
                 'Transition zone % core involvement': tz_p_inv_r,
                 'Age': user_input['Age'],
                 'Worst Gleason Grade Group': sort_g_p_inv_r['Gleason'].max(),
                 'Mid % core involvement': mid_p_inv_r
                 }

    # Save positive cores and cores taken at each site for annotated prostate diagram later
    get_user_input_r.b_findings_r = base_findings_r
    get_user_input_r.b_p_core_r = base_p_core_r
    get_user_input_r.b_t_core_r = base_t_core_r
    get_user_input_r.b_p_r = base_p_inv_r
    get_user_input_r.m_findings_r = mid_findings_r
    get_user_input_r.m_p_core_r = mid_p_core_r
    get_user_input_r.m_t_core_r = mid_t_core_r
    get_user_input_r.m_p_r = mid_p_inv_r
    get_user_input_r.a_findings_r = apex_findings_r
    get_user_input_r.a_p_core_r = apex_p_core_r
    get_user_input_r.a_t_core_r = apex_t_core_r
    get_user_input_r.a_p_r = apex_p_inv_r
    get_user_input_r.t_findings_r = tz_findings_r
    get_user_input_r.t_p_core_r = tz_p_core_r
    get_user_input_r.t_t_core_r = tz_t_core_r
    get_user_input_r.t_p_r = tz_p_inv_r

    pt_features_r = pd.DataFrame(pt_data_r, index=[0])
    return pt_features_r


user_input_r = get_user_input_r()

# Store the model predictions as a variable
# = model.predict_proba(user_input)
# prediction_r = model.predict_proba(user_input_r)

# Create 2 columns, one to show SHAP plots, one to show annotated prostate diagram
col1, col2 = st.beta_columns([1, 1.75])

# SHAP plots under column 2
col2.header('Model explanations')
col2.write('The probability of ssEPE for each lobe is indicated in **bold**. \
Each plot highlights which features have the greatest impact on the predicted probability of ssEPE')

# SHAP plot for left lobe
col2.subheader('Left lobe')
st.set_option('deprecation.showPyplotGlobalUse', False)
# shap.initjs()
shap_values = explainer.shap_values(user_input)
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
shap.force_plot(0.3, shap_values, user_input, features_list, text_rotation=10,  # features_list,
                matplotlib=True)
col2.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')
plt.clf()

# SHAP plot for right lobe
col2.subheader('Right lobe')
shap_values_r = explainer.shap_values(user_input_r)
features_list_r = ('PSA',
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
shap.plots.force(0.3, shap_values_r, user_input_r, features_list_r, matplotlib=True,
                text_rotation=10)
col2.pyplot(bbox_inches='tight', dpi=600, pad_inches=0, use_column_width='auto')
plt.clf()

# Show annotated prostate diagram under column 2
# Importing Image and ImageFont, ImageDraw module from PIL package
col1.header('Annotated Prostate')
col1.write('Automatically updates based on user-entered values')

# Specify font size for annotated prostate diagram
font = ImageFont.truetype('arial.ttf', 50)

# Create text to overlay on annotated prostate diagram, auto-updates based on user inputted values
base_L = str(G_CHOICES[get_user_input.b_findings]) + '\n' \
         + 'Positive cores: ' + str(get_user_input.b_p_core) + '/' + str(get_user_input.b_t_core) + '\n' \
         + '% core inv: ' + str(get_user_input.b_p)
mid_L = str(G_CHOICES[get_user_input.m_findings]) + '\n' \
        + 'Positive cores: ' + str(get_user_input.m_p_core) + '/' + str(get_user_input.m_t_core) + '\n' \
        + '% core inv: ' + str(get_user_input.m_p)
apex_L = str(G_CHOICES[get_user_input.a_findings]) + '\n' \
         + 'Positive cores: ' + str(get_user_input.a_p_core) + '/' + str(get_user_input.a_t_core) + '\n' \
         + '% core inv: ' + str(get_user_input.a_p)
tz_L = str(G_CHOICES[get_user_input.t_findings]) + '\n' \
       + 'Positive cores: ' + str(get_user_input.t_p_core) + '/' + str(get_user_input.t_t_core) + '\n' \
       + '% core inv: ' + str(get_user_input.t_p)
base_R = str(G_CHOICES[get_user_input_r.b_findings_r]) + '\n' \
         + 'Positive cores: ' + str(get_user_input_r.b_p_core_r) + '/' + str(get_user_input_r.b_t_core_r) + '\n' \
         + '% core inv: ' + str(get_user_input_r.b_p_r)
mid_R = str(G_CHOICES[get_user_input_r.m_findings_r]) + '\n' \
        + 'Positive cores: ' + str(get_user_input_r.m_p_core_r) + '/' + str(get_user_input_r.m_t_core_r) + '\n' \
        + '% core inv: ' + str(get_user_input_r.m_p_r)
apex_R = str(G_CHOICES[get_user_input_r.a_findings_r]) + '\n' \
         + 'Positive cores: ' + str(get_user_input_r.a_p_core_r) + '/' + str(get_user_input_r.a_t_core_r) + '\n' \
         + '% core inv: ' + str(get_user_input_r.a_p_r)
tz_R = str(G_CHOICES[get_user_input_r.t_findings_r]) + '\n' \
       + 'Positive cores: ' + str(get_user_input_r.t_p_core_r) + '/' + str(get_user_input_r.t_t_core_r) + '\n' \
       + '% core inv: ' + str(get_user_input_r.t_p_r)

# Set conditions to show colour coded site images based on Gleason Grade Group for each site
draw = ImageDraw.Draw(image2)
if get_user_input.b_findings == 3:
    image2.paste(image_bl_G1, (145, 958), mask=image_bl_G1)
if get_user_input.b_findings == 4:
    image2.paste(image_bl_G2, (145, 958), mask=image_bl_G2)
if get_user_input.b_findings == 5:
    image2.paste(image_bl_G3, (145, 958), mask=image_bl_G3)
if get_user_input.b_findings == 6:
    image2.paste(image_bl_G4, (145, 958), mask=image_bl_G4)
if get_user_input.b_findings == 7:
    image2.paste(image_bl_G5, (145, 958), mask=image_bl_G5)

if get_user_input.m_findings == 3:
    image2.paste(image_ml_G1, (145, 606), mask=image_ml_G1)
if get_user_input.m_findings == 4:
    image2.paste(image_ml_G2, (145, 606), mask=image_ml_G2)
if get_user_input.m_findings == 5:
    image2.paste(image_ml_G3, (145, 606), mask=image_ml_G3)
if get_user_input.m_findings == 6:
    image2.paste(image_ml_G4, (145, 606), mask=image_ml_G4)
if get_user_input.m_findings == 7:
    image2.paste(image_ml_G5, (145, 606), mask=image_ml_G5)

if get_user_input.a_findings == 3:
    image2.paste(image_al_G1, (145, 130), mask=image_al_G1)
if get_user_input.a_findings == 4:
    image2.paste(image_al_G2, (145, 130), mask=image_al_G2)
if get_user_input.a_findings == 5:
    image2.paste(image_al_G3, (145, 130), mask=image_al_G3)
if get_user_input.a_findings == 6:
    image2.paste(image_al_G4, (145, 130), mask=image_al_G4)
if get_user_input.a_findings == 7:
    image2.paste(image_al_G5, (145, 130), mask=image_al_G5)

if get_user_input.t_findings == 3:
    image2.paste(image_tl_G1, (665, 493), mask=image_tl_G1)
if get_user_input.t_findings == 4:
    image2.paste(image_tl_G2, (665, 493), mask=image_tl_G2)
if get_user_input.t_findings == 5:
    image2.paste(image_tl_G3, (665, 493), mask=image_tl_G3)
if get_user_input.t_findings == 6:
    image2.paste(image_tl_G4, (665, 493), mask=image_tl_G4)
if get_user_input.t_findings == 7:
    image2.paste(image_tl_G5, (665, 493), mask=image_tl_G5)

if get_user_input_r.b_findings_r == 3:
    image2.paste(image_br_G1, (1104, 958), mask=image_br_G1)
if get_user_input_r.b_findings_r == 4:
    image2.paste(image_br_G2, (1104, 958), mask=image_br_G2)
if get_user_input_r.b_findings_r == 5:
    image2.paste(image_br_G3, (1104, 958), mask=image_br_G3)
if get_user_input_r.b_findings_r == 6:
    image2.paste(image_br_G4, (1104, 958), mask=image_br_G4)
if get_user_input_r.b_findings_r == 7:
    image2.paste(image_br_G5, (1104, 958), mask=image_br_G5)

if get_user_input_r.m_findings_r == 3:
    image2.paste(image_mr_G1, (1542, 606), mask=image_mr_G1)
if get_user_input_r.m_findings_r == 4:
    image2.paste(image_mr_G2, (1542, 606), mask=image_mr_G2)
if get_user_input_r.m_findings_r == 5:
    image2.paste(image_mr_G3, (1542, 606), mask=image_mr_G3)
if get_user_input_r.m_findings_r == 6:
    image2.paste(image_mr_G4, (1542, 606), mask=image_mr_G4)
if get_user_input_r.m_findings_r == 7:
    image2.paste(image_mr_G5, (1542, 606), mask=image_mr_G5)

if get_user_input_r.a_findings_r == 3:
    image2.paste(image_ar_G1, (1104, 130), mask=image_ar_G1)
if get_user_input_r.a_findings_r == 4:
    image2.paste(image_ar_G2, (1104, 130), mask=image_ar_G2)
if get_user_input_r.a_findings_r == 5:
    image2.paste(image_ar_G3, (1104, 130), mask=image_ar_G3)
if get_user_input_r.a_findings_r == 6:
    image2.paste(image_ar_G4, (1104, 130), mask=image_ar_G4)
if get_user_input_r.a_findings_r == 7:
    image2.paste(image_ar_G5, (1104, 130), mask=image_ar_G5)

if get_user_input_r.t_findings_r == 3:
    image2.paste(image_tr_G1, (1100, 493), mask=image_tr_G1)
if get_user_input_r.t_findings_r == 4:
    image2.paste(image_tr_G2, (1100, 493), mask=image_tr_G2)
if get_user_input_r.t_findings_r == 5:
    image2.paste(image_tr_G3, (1100, 493), mask=image_tr_G3)
if get_user_input_r.t_findings_r == 6:
    image2.paste(image_tr_G4, (1100, 493), mask=image_tr_G4)
if get_user_input_r.t_findings_r == 7:
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
           '**Width of bars**: Importance of the feature. The wider it is, the greater impact it has on risk of ssEPE')

st.write('See where you fit among the study population')
col_left, col_right = st.beta_columns([1, 1])
col_left.radio("Select left lobe feature to compare", user_input)

# Display additional text

with st.beta_expander("See how the model was developed"):
    st.write("""""")
    st.write('A retrospective sample of 900 prostatic lobes (450 patients) from radical prostatectomy (RP) specimens at\
     Credit Valley Hospital, Mississauga, between 2010 and 2020, was used as the training cohort. Features\
     (ie: variables) included patient demographics, clinical, and site-specific data from\
     transrectal ultrasound-guided prostate biopsy. The primary label (ie: outcome) of interest was the presence\
     of EPE in the ipsilateral lobe of the prostatectomy specimen. A previously developed [model]\
    (https://bjui-journals.onlinelibrary.wiley.com/doi/full/10.1111/bju.13733), which has the highest performance out of\
     current biopsy-derived predictive models for ssEPE that have been externally validated,\
      was used as the baseline model for comparison.')
    st.write('Dimensionality reduction was performed using a modified [Boruta](https://www.jstatsoft.org/article/view/v036i11/0)\
      algorithm followed by removing highly correlated features (Pearson correlation > 0.8). This former involves\
       fitting all features to a random forest model and determining feature importance\
       by comparing the relevance of each feature to that of random noise. Given that our dataset contains both\
        categorical and numerical features, SHAP was specifically selected in lieu of impurity-based measures\
         to reduce bias towards high cardinality features.')
    st.write('Using the final set of the most important and independent features, a stratified ten-fold \
     cross-validation method was performed to train a gradient-boosted machine, optimize hyperparameters,\
      and for internal validation. In stratified cross-validation, the training cohort was randomly partitioned\
       into ten equal folds, with each fold containing the same percentage of positive ssEPE cases. Nine folds\
        were used for model training and hyperparameter tuning while the remaining fold made up the validation cohort.\
         This process was repeated ten times such that each fold served as the validation cohort once. Model\
          performance was determined based on the average performance across all ten validation cohorts to improve\
           generalizability of the models. All models were further externally validated using a testing cohort of\
            122 lobes (61 patients) from RP specimens at Mississauga Hospital, Mississauga, between 2016 and 2020.\
            All models were assessed by area under receiver-operating-characteristic curve (AUROC),\
             precision-recall curve (AUPRC), calibration curve, and decision curve analysis. The incidence of ssEPE in\
              the training and testing cohorts were 30.7 and 41.8%, respectively.')

    st.write("""""")
    st.write("""""")
    colF, colG, colZ = st.beta_columns([1, 1, 1])
    colF.write('**Area under receiver-operating-characteristic curve (AUROC):** is used to measure the discriminative\
               capability of predictive models by comparing the true positive rate (sensitivity) and false positive\
               rate (1-specificity) across various decision thresholds.')
    colF.write('Our ML model outperformed the baseline model with a **mean AUROC of 0.81** (95% CI 0.78-0.83) **vs \
                0.74** (95% CI 0.71-0.76) **(p<0.01)** on cross-validation of the training cohort. Similarly, our ML\
                 model performed favourably on the external testing cohort with an **AUROC of 0.81** (95% CI 0.73-0.88)\
                  **vs 0.75** (95% CI 0.67-0.83) **(p=0.03)**.')
    colG.image(auroc_train, use_column_width='auto')
    colZ.image(auroc_test, use_column_width='auto')
    st.write("""""")
    st.write("""""")
    colH, colI, colY = st.beta_columns([1, 1, 1])
    colH.write('**Area under precision-recall curve (AUPRC):** compares precision (positive predictive value) and \
               recall (sensitivity) across various decision thresholds. It is more informative than AUROC curves when\
               evaluating the performance of classifiers for imbalanced datasets, such as in our case where there are\
                more patients without ssEPE than with ssEPE. This is because AUPRC evaluates the proportion of true\
               positives among positive predictions, which is our outcome of interest.')
    colH.write('Our ML model outperformed the baseline model with a **mean AUPRC of 0.69** (95% CI 0.64-0.73) **vs \
                0.59** (95% CI 0.54-0.65) on cross-validation of the training cohort. Similarly, our ML model \
                performed favourably on the external testing cohort with an **AUPRC of 0.78** (95% CI 0.67-0.86) **vs \
                0.70** (95% CI 0.60-0.79).')
    colI.image(auprc_train, use_column_width='auto')
    colY.image(auprc_test, use_column_width='auto')
    st.write("""""")
    st.write("""""")
    colJ, colK, colX = st.beta_columns([1, 1, 1])
    colJ.write('**Calibration curves:** are used to evaluate the accuracy of model risk estimates by measuring the\
               agreement between the predicted and observed number of outcomes. A perfectly calibrated model is\
               depicted as a 45 degree line. In our case, if a calibration curve is above the reference line, it\
                underestimates the risk of ssEPE, which may lead to undertreatment (ie: leaving some cancer behind).\
                 However, if a calibration curve is below the reference line, it overestimates the risk of ssEPE, which\
                  may lead to overtreatment (ie: patient gets unnecessarily treated with a non-nerve sparing approach).\
                   Therefore, calibration is especially important when evaluating predictive models used to support\
                    decision-making.')
    colJ.write('Our ML model is well calibrated for predicted probabilities between 0-40%, while overestimating the risk\
                of ssEPE above 40% probability in the testing cohort.')
    colK.image(calib_train, use_column_width='auto')
    colX.image(calib_test, use_column_width='auto')
    st.write("""""")
    st.write("""""")
    colL, colM = st.beta_columns([1, 1])
    colL.write('**[Decision curve analysis](https://pubmed.ncbi.nlm.nih.gov/17099194/):** is used to evaluate clinical\
                utility. Here, the net benefit of the model is plotted against various threshold probabilities for\
                 three different treatment strategies: treat all, treat none, or treat only those predicted to have\
                  ssEPE by the model.')
    colL.write('Threshold probabilities between 10-30% were deemed the most clinically relevant for consideration\
                 of nerve-sparing. Our ML model achieved a higher net benefit than the baseline model across these\
                 thresholds. This translates to a potential **increase in appropriate nerve-sparing by 20 vs\
                 8 per 100 cases at a threshold probability of 20%** compared to a "treat all" strategy.')
    colM.image(dca, use_column_width='auto')

    st.write("""""")
    st.write("""""")
    st.write('This model was developed in accordance to the STREAM-URO framework (see table below).')
    st.write("""""")
    st.image(stream_uro, width=800)
    st.write("""""")

with st.beta_expander("See how the model explanations were determined"):
    st.write("""""")
    st.write('Model explanations were calculated based on SHAP (SHapley Additive exPlanations) values,\
    originally developed by [Lundberg and Lee (2017)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf).\
    This is an additive feature attribution method that satisfies all three properties of explanation models: local accuracy, missingness, and consistency.')
    st.write('**Accuracy**: the output of the explanation model must match the output of the original model for a\
     given prediction.  \n'
             '**Missingness**: when a feature is missing, it should have no impact on the model.  \n'
             '**Consistency**: a feature’s assigned attribution must match its degree of importance in the original model\
    (ie: if overall – % tissue involvement has the highest attribution, it must also have the highest feature importance\
    and the model must rely on this feature the most for a given prediction).')
    st.write('SHAP allows us to understand why our model made a given prediction by simplifying our complex model into a\
    linear function of binary variables. This approach has previously been implemented to improve understanding\
    of [hypoxemia risk during anesthetic care](https://www.nature.com/articles/s41551-018-0304-0).')
    st.write("""""")

with st.beta_expander("Additional model explanations"):
    st.write("""""")
    colA, colB = st.beta_columns([1, 1])
    colA.write("**Feature importance rankings:** helps identify which features had the overall greatest impact on\
             our ML model's predictions. Here, we see that PSA, Maximum % core involvement, and perineural invasion\
             were the three most important features in our ML model.")
    colB.image(summary, use_column_width='auto')
    st.write("""""")
    st.write("""""")
    colD, colE = st.beta_columns([1, 3])
    colD.write('**Partial dependence plots:** allows us to visualize how a given feature can impact the probability of \
             ssEPE across all its possible values (ie: how does % Gleason pattern 4/5, from 0 to 100%, positively or\
              negatively impact probability of ssEPE?). We see that our ML model represents each feature in different\
             ways. Some have a linear or logarithmic relationship, while others are more complex.')
    colE.image(pdp, use_column_width='auto')

st.write("""""")
st.write("""""")


st.write('Developed by: Jethro CC. Kwong$^{1,2}$, Adree Khondker$^{3}$, Christopher Tran$^{3}$, Emily Evans$^{3}$,\
 Amna Ali$^{4}$, Munir Jamal$^{1}$, Thomas Short$^{1}$, Frank Papanikolaou$^{1}$,\
  John R. Srigley$^{5}$, Andrew H. Feifer$^{1,4}$')


st.write('$^{1}$Division of Urology, Department of Surgery, University of Toronto, Toronto, ON, Canada  \n'
         '$^{2}$Temerty Centre for AI Research and Education in Medicine, University of Toronto, Toronto, Canada  \n'
         '$^{3}$Temerty Faculty of Medicine, University of Toronto, Toronto, ON, Canada  \n'
         '$^{4}$Institute for Better Health, Trillium Health Partners, Mississauga, ON, Canada  \n'
         '$^{5}$Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, ON, Canada')

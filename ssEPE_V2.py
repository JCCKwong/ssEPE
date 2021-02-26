# Description: This program predicts risk of side-specific extraprostatic extension

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import PIL.Image
import streamlit as st
import shap
import joblib
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import ImageFont, ImageDraw, ImageOps
# Import ML model of choice
import xgboost as xgb

# Default widescreen mode
def _max_width_():
    max_width_str = f"max-width: 2000px;"
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
st.title('Side-specific extraprostatic extension (EPE) prediction')
st.write('Determine probability of EPE in ipsilateral lobe using clinicopathological features and machine learning')
st.write('Developed by: Jethro CC Kwong, Adree Khondker, Christopher Tran, Emily Evans, Amna Ali, Munir Jamal,\
 Thomas Short, Frank Papanikolaou, John R. Srigley, Andrew H. Feifer')

## LOAD TRAINED RANDOM FOREST MODEL
cloud_model_location = '1WspVBYOLjmQHPusl6I8f2LkG_bO9QtG7'  # hosted on GD
cloud_explainer_location = '1WspVBYOLjmQHPusl6I8f2LkG_bO9QtG7'  # hosted on GD


@st.cache(allow_output_mutation=True)
def load_model():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    f_checkpoint = Path('XGB ssEPE model V3.pkl')
    f_checkpoint1 = Path('Features.pkl')
    # download from GD if model or explainer not present
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdd.download_file_from_google_drive(cloud_model_location, f_checkpoint)
    if not f_checkpoint1.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdd.download_file_from_google_drive(cloud_explainer_location, f_checkpoint1)

    model = joblib.load(f_checkpoint)
    features = joblib.load(f_checkpoint1)
    explainer = shap.TreeExplainer(model, features, model_output='probability')
    return model, explainer

model, explainer = load_model()

def load_images():
    # Load blank prostate and all colour coded sites as image objects
    image2 = PIL.Image.open('Prostate diagram.png')
    image_bl_G1 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason1.png')))
    image_bl_G2 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason2.png')))
    image_bl_G3 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason3.png')))
    image_bl_G4 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason4.png')))
    image_bl_G5 = PIL.ImageOps.flip(PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason5.png')))
    image_ml_G1 = PIL.Image.open('Mid_Gleason1.png')
    image_ml_G2 = PIL.Image.open('Mid_Gleason2.png')
    image_ml_G3 = PIL.Image.open('Mid_Gleason3.png')
    image_ml_G4 = PIL.Image.open('Mid_Gleason4.png')
    image_ml_G5 = PIL.Image.open('Mid_Gleason5.png')
    image_al_G1 = PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason1.png'))
    image_al_G2 = PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason2.png'))
    image_al_G3 = PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason3.png'))
    image_al_G4 = PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason4.png'))
    image_al_G5 = PIL.ImageOps.mirror(PIL.Image.open('Corner_Gleason5.png'))
    image_tl_G1 = PIL.Image.open('TZ_Gleason1.png')
    image_tl_G2 = PIL.Image.open('TZ_Gleason2.png')
    image_tl_G3 = PIL.Image.open('TZ_Gleason3.png')
    image_tl_G4 = PIL.Image.open('TZ_Gleason4.png')
    image_tl_G5 = PIL.Image.open('TZ_Gleason5.png')
    image_br_G1 = PIL.ImageOps.flip(PIL.Image.open('Corner_Gleason1.png'))
    image_br_G2 = PIL.ImageOps.flip(PIL.Image.open('Corner_Gleason2.png'))
    image_br_G3 = PIL.ImageOps.flip(PIL.Image.open('Corner_Gleason3.png'))
    image_br_G4 = PIL.ImageOps.flip(PIL.Image.open('Corner_Gleason4.png'))
    image_br_G5 = PIL.ImageOps.flip(PIL.Image.open('Corner_Gleason5.png'))
    image_mr_G1 = PIL.Image.open('Mid_Gleason1.png')
    image_mr_G2 = PIL.Image.open('Mid_Gleason2.png')
    image_mr_G3 = PIL.Image.open('Mid_Gleason3.png')
    image_mr_G4 = PIL.Image.open('Mid_Gleason4.png')
    image_mr_G5 = PIL.Image.open('Mid_Gleason5.png')
    image_ar_G1 = PIL.Image.open('Corner_Gleason1.png')
    image_ar_G2 = PIL.Image.open('Corner_Gleason2.png')
    image_ar_G3 = PIL.Image.open('Corner_Gleason3.png')
    image_ar_G4 = PIL.Image.open('Corner_Gleason4.png')
    image_ar_G5 = PIL.Image.open('Corner_Gleason5.png')
    image_tr_G1 = PIL.ImageOps.mirror(PIL.Image.open('TZ_Gleason1.png'))
    image_tr_G2 = PIL.ImageOps.mirror(PIL.Image.open('TZ_Gleason2.png'))
    image_tr_G3 = PIL.ImageOps.mirror(PIL.Image.open('TZ_Gleason3.png'))
    image_tr_G4 = PIL.ImageOps.mirror(PIL.Image.open('TZ_Gleason4.png'))
    image_tr_G5 = PIL.ImageOps.mirror(PIL.Image.open('TZ_Gleason5.png'))
    return image2, image_bl_G1, image_bl_G2, image_bl_G3, image_bl_G4, image_bl_G5,\
    image_ml_G1, image_ml_G2, image_ml_G3, image_ml_G4, image_ml_G5,\
    image_al_G1, image_al_G2, image_al_G3, image_al_G4, image_al_G5,\
    image_tl_G1, image_tl_G2, image_tl_G3, image_tl_G4, image_tl_G5,\
    image_br_G1, image_br_G2, image_br_G3, image_br_G4, image_br_G5,\
    image_mr_G1, image_mr_G2, image_mr_G3, image_mr_G4, image_mr_G5,\
    image_ar_G1, image_ar_G2, image_ar_G3, image_ar_G4, image_ar_G5,\
    image_tr_G1, image_tr_G2, image_tr_G3, image_tr_G4, image_tr_G5

image2, image_bl_G1, image_bl_G2, image_bl_G3, image_bl_G4, image_bl_G5,\
    image_ml_G1, image_ml_G2, image_ml_G3, image_ml_G4, image_ml_G5,\
    image_al_G1, image_al_G2, image_al_G3, image_al_G4, image_al_G5,\
    image_tl_G1, image_tl_G2, image_tl_G3, image_tl_G4, image_tl_G5,\
    image_br_G1, image_br_G2, image_br_G3, image_br_G4, image_br_G5,\
    image_mr_G1, image_mr_G2, image_mr_G3, image_mr_G4, image_mr_G5,\
    image_ar_G1, image_ar_G2, image_ar_G3, image_ar_G4, image_ar_G5,\
    image_tr_G1, image_tr_G2, image_tr_G3, image_tr_G4, image_tr_G5 = load_images()

# Define choices and labels for feature inputs
CHOICES = {0: 'No', 1: 'Yes'}
def format_func_yn(option):
    return CHOICES[option]

G_CHOICES = {0: 'Normal', 1: 'HGPIN', 2: 'ASAP', 3: 'Gleason 3+3', 4: 'Gleason 3+4', 5: 'Gleason 4+3', 6: 'Gleason 4+4', 7: 'Gleason 4+5/5+4'}
def format_func_gleason(option):
    return G_CHOICES[option]

# Create sidebar for user inputted values
st.sidebar.write('Enter patient values')

# Create sidebar inputs for global variables and left lobe variables
def get_user_input():
    with st.sidebar.beta_expander('Global variables',expanded=True):
        age = st.number_input('Age (years)', 0.0, 100.0)
        psa = st.number_input('PSA (ng/ml)', 0.00, 200.00)
        p_high = st.slider('% Gleason pattern 4/5', 0.0, 100.00, 0.0, 0.5)
        perineural_inv = st.selectbox('Perineural invasion', options=list(CHOICES.keys()), format_func=format_func_yn)
    with st.sidebar.beta_expander('Side-specific variables (Left)',expanded=True):
        base_findings = st.selectbox('Base findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                     key=0)
        base_p_core = st.number_input('Base # of positive cores', 0, 10, value=0, key=0)
        base_t_core = st.number_input('Base # of cores taken', 0, 10, value=2, key=0)
        base_p_inv = st.number_input('Base % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=0)
        mid_findings = st.selectbox('Mid findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                    key=0)
        mid_p_core = st.number_input('Mid # of positive cores', 0, 10, value=0, key=0)
        mid_t_core = st.number_input('Mid # of cores taken', 0, 10, value=2, key=0)
        mid_p_inv = st.number_input('Mid % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=0)
        apex_findings = st.selectbox('Apex findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                     key=0)
        apex_p_core = st.number_input('Apex # of positive cores', 0, 10, value=0, key=0)
        apex_t_core = st.number_input('Apex # of cores taken', 0, 10, value=1, key=0)
        apex_p_inv = st.number_input('Apex % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=0)
        tz_findings = st.selectbox('Transition zone findings', options=list(G_CHOICES.keys()),
                                   format_func=format_func_gleason, key=0)
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
    pt_data = {'Age': age,
               'PSA': psa,
               '% Gleason pattern 4/5': p_high,
               'Perineural invasion': perineural_inv,
               '% core involvement': p_core_total / t_core_total,
               'Worst Gleason': sort_g_p_inv['Gleason'].max(),
               'Maximum % core involvement': sort_g_p_inv['% core involvement'].max(),
               'Base findings': base_findings,
               'Base % core involvement': base_p_inv,
               'Mid % core involvement': mid_p_inv,
               'Transition zone % core involvement': tz_p_inv
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

    pt_features = pd.DataFrame(pt_data, index = [0])
    return pt_features

# Store the left lobe user input into a variable
user_input = get_user_input()

def get_user_input_r():
    with st.sidebar.beta_expander('Side-specific variables (Right)', expanded=True):
        base_findings_r = st.selectbox('Base findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                       key=1)
        base_p_core_r = st.number_input('Base # of positive cores', 0, 10, value=0, key=1)
        base_t_core_r = st.number_input('Base # of cores taken', 0, 10, value=2, key=1)
        base_p_inv_r = st.number_input('Base % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=1)
        mid_findings_r = st.selectbox('Mid findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                      key=1)
        mid_p_core_r = st.number_input('Mid # of positive cores', 0, 10, value=0, key=1)
        mid_t_core_r = st.number_input('Mid # of cores taken', 0, 10, value=2, key=1)
        mid_p_inv_r = st.number_input('Mid % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=1)
        apex_findings_r = st.selectbox('Apex findings', options=list(G_CHOICES.keys()), format_func=format_func_gleason,
                                       key=1)
        apex_p_core_r = st.number_input('Apex # of positive cores', 0, 10, value=0, key=1)
        apex_t_core_r = st.number_input('Apex # of cores taken', 0, 10, value=1, key=1)
        apex_p_inv_r = st.number_input('Apex % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=1)
        tz_findings_r = st.selectbox('Transition zone findings', options=list(G_CHOICES.keys()),
                                     format_func=format_func_gleason, key=1)
        tz_p_core_r = st.number_input('Transition zone # of positive cores', 0, 10, value=0, key=1)
        tz_t_core_r = st.number_input('Transition zone # of cores taken', 0, 10, value=1, key=1)
        tz_p_inv_r = st.number_input('Transition zone % core involvement (0 to 100)', 0.0, 100.0, value=0.0, key=1)

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
    pt_data_r = {'Age': user_input['Age'],
                 'PSA': user_input['PSA'],
                 '% Gleason pattern 4/5': user_input['% Gleason pattern 4/5'],
                 'Perineural invasion': user_input['Perineural invasion'],
                 '% core involvement': p_core_total_r / t_core_total_r,
                 'Worst Gleason': sort_g_p_inv_r['Gleason'].max(),
                 'Maximum % core involvement': sort_g_p_inv_r['% core involvement'].max(),
                 'Base findings': base_findings_r,
                 'Base % core involvement': base_p_inv_r,
                 'Mid % core involvement': mid_p_inv_r,
                 'Transition zone % core involvement': tz_p_inv_r
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

    pt_features_r = pd.DataFrame(pt_data_r, index = [0])
    return pt_features_r

user_input_r = get_user_input_r()

# Store the model predictions as a variable
prediction = model.predict_proba(user_input)
prediction_r = model.predict_proba(user_input_r)

# Create 2 columns, one to show SHAP plots, one to show annotated prostate diagram
col1, col2 = st.beta_columns([1, 1.75])

# SHAP plots under column 2
col2.header('Model explanations')
col2.write('Highlights which features have the greatest impact on the predicted probability of EPE')

# SHAP plot for left lobe
col2.subheader('Left lobe')
st.set_option('deprecation.showPyplotGlobalUse', False)
#shap.initjs()
shap_values = explainer.shap_values(user_input)
features_list = ('Age',
                 'PSA',
                 '% Gleason pattern 4/5',
                 'Perineural invasion',
                 '% core involvement',
                 'Worst Gleason',
                 'Maximum % core involvement',
                 'Base findings',
                 'Base % core involvement',
                 'Mid % core involvement',
                 'Transition zone % core involvement')
shap.force_plot(explainer.expected_value, shap_values, user_input, features_list, matplotlib=True, text_rotation=10)
col2.pyplot(bbox_inches='tight', dpi=600, pad_inches=0)
plt.clf()

# SHAP plot for right lobe
col2.subheader('Right lobe')
shap_values_r = explainer.shap_values(user_input_r)
features_list_r = ('Age',
                   'PSA',
                   '% Gleason pattern 4/5',
                   'Perineural invasion',
                   '% core involvement',
                   'Worst Gleason',
                   'Maximum % core involvement',
                   'Base findings',
                   'Base % core involvement',
                   'Mid % core involvement',
                   'Transition zone % core involvement')
shap.force_plot(explainer.expected_value, shap_values_r, user_input_r, features_list_r, matplotlib=True, text_rotation=10)
col2.pyplot(bbox_inches='tight', dpi=600, pad_inches=0)
plt.clf()

# Show annotated prostate diagram under column 2
# Importing Image and ImageFont, ImageDraw module from PIL package
col1.header('Annotated Prostate')
col1.write('Automatically updates based on user entered values')

# Specify font size for annotated prostate diagram
font = ImageFont.truetype('arial.ttf', 50)

# Create text to overlay on annotated prostate diagram, auto-updates based on user inputted values
base_L = str(G_CHOICES[get_user_input.b_findings]) + '\n'\
         + 'Positive cores: ' + str(get_user_input.b_p_core) + '/' + str(get_user_input.b_t_core) + '\n'\
         + '% core inv: ' + str(get_user_input.b_p)
mid_L = str(G_CHOICES[get_user_input.m_findings]) + '\n'\
         + 'Positive cores: ' + str(get_user_input.m_p_core) + '/' + str(get_user_input.m_t_core) + '\n'\
         + '% core inv: ' + str(get_user_input.m_p)
apex_L = str(G_CHOICES[get_user_input.a_findings]) + '\n'\
         + 'Positive cores: ' + str(get_user_input.a_p_core) + '/' + str(get_user_input.a_t_core) + '\n'\
         + '% core inv: ' + str(get_user_input.a_p)
tz_L = str(G_CHOICES[get_user_input.t_findings]) + '\n'\
         + 'Positive cores: ' + str(get_user_input.t_p_core) + '/' + str(get_user_input.t_t_core) + '\n'\
         + '% core inv: ' + str(get_user_input.t_p)
base_R = str(G_CHOICES[get_user_input_r.b_findings_r]) + '\n'\
         + 'Positive cores: ' + str(get_user_input_r.b_p_core_r) + '/' + str(get_user_input_r.b_t_core_r) + '\n'\
         + '% core inv: ' + str(get_user_input_r.b_p_r)
mid_R = str(G_CHOICES[get_user_input_r.m_findings_r]) + '\n'\
         + 'Positive cores: ' + str(get_user_input_r.m_p_core_r) + '/' + str(get_user_input_r.m_t_core_r) + '\n'\
         + '% core inv: ' + str(get_user_input_r.m_p_r)
apex_R = str(G_CHOICES[get_user_input_r.a_findings_r]) + '\n'\
         + 'Positive cores: ' + str(get_user_input_r.a_p_core_r) + '/' + str(get_user_input_r.a_t_core_r) + '\n'\
         + '% core inv: ' + str(get_user_input_r.a_p_r)
tz_R = str(G_CHOICES[get_user_input_r.t_findings_r]) + '\n'\
         + 'Positive cores: ' + str(get_user_input_r.t_p_core_r) + '/' + str(get_user_input_r.t_t_core_r) + '\n'\
         + '% core inv: ' + str(get_user_input_r.t_p_r)

# Set conditions to show colour coded site images based on Gleason Grade Group for each site
draw = ImageDraw.Draw(image2)
if get_user_input.b_findings==3:
    image2.paste(image_bl_G1, (145, 958), mask=image_bl_G1)
if get_user_input.b_findings==4:
    image2.paste(image_bl_G2, (145, 958), mask=image_bl_G2)
if get_user_input.b_findings==5:
    image2.paste(image_bl_G3, (145, 958), mask=image_bl_G3)
if get_user_input.b_findings==6:
    image2.paste(image_bl_G4, (145, 958), mask=image_bl_G4)
if get_user_input.b_findings==7:
    image2.paste(image_bl_G5, (145, 958), mask=image_bl_G5)

if get_user_input.m_findings==3:
    image2.paste(image_ml_G1, (145, 606), mask=image_ml_G1)
if get_user_input.m_findings==4:
    image2.paste(image_ml_G2, (145, 606), mask=image_ml_G2)
if get_user_input.m_findings==5:
    image2.paste(image_ml_G3, (145, 606), mask=image_ml_G3)
if get_user_input.m_findings==6:
    image2.paste(image_ml_G4, (145, 606), mask=image_ml_G4)
if get_user_input.m_findings==7:
    image2.paste(image_ml_G5, (145, 606), mask=image_ml_G5)

if get_user_input.a_findings==3:
    image2.paste(image_al_G1, (145, 130), mask=image_al_G1)
if get_user_input.a_findings==4:
    image2.paste(image_al_G2, (145, 130), mask=image_al_G2)
if get_user_input.a_findings==5:
    image2.paste(image_al_G3, (145, 130), mask=image_al_G3)
if get_user_input.a_findings==6:
    image2.paste(image_al_G4, (145, 130), mask=image_al_G4)
if get_user_input.a_findings==7:
    image2.paste(image_al_G5, (145, 130), mask=image_al_G5)

if get_user_input.t_findings==3:
    image2.paste(image_tl_G1, (665, 493), mask=image_tl_G1)
if get_user_input.t_findings==4:
    image2.paste(image_tl_G2, (665, 493), mask=image_tl_G2)
if get_user_input.t_findings==5:
    image2.paste(image_tl_G3, (665, 493), mask=image_tl_G3)
if get_user_input.t_findings==6:
    image2.paste(image_tl_G4, (665, 493), mask=image_tl_G4)
if get_user_input.t_findings==7:
    image2.paste(image_tl_G5, (665, 493), mask=image_tl_G5)

if get_user_input_r.b_findings_r==3:
    image2.paste(image_br_G1,(1104,958), mask=image_br_G1)
if get_user_input_r.b_findings_r==4:
    image2.paste(image_br_G2,(1104,958), mask=image_br_G2)
if get_user_input_r.b_findings_r==5:
    image2.paste(image_br_G3,(1104,958), mask=image_br_G3)
if get_user_input_r.b_findings_r==6:
    image2.paste(image_br_G4,(1104,958), mask=image_br_G4)
if get_user_input_r.b_findings_r==7:
    image2.paste(image_br_G5,(1104,958), mask=image_br_G5)

if get_user_input_r.m_findings_r==3:
    image2.paste(image_mr_G1, (1542, 606), mask=image_mr_G1)
if get_user_input_r.m_findings_r==4:
    image2.paste(image_mr_G2, (1542, 606), mask=image_mr_G2)
if get_user_input_r.m_findings_r==5:
    image2.paste(image_mr_G3, (1542, 606), mask=image_mr_G3)
if get_user_input_r.m_findings_r==6:
    image2.paste(image_mr_G4, (1542, 606), mask=image_mr_G4)
if get_user_input_r.m_findings_r==7:
    image2.paste(image_mr_G5, (1542, 606), mask=image_mr_G5)

if get_user_input_r.a_findings_r==3:
    image2.paste(image_ar_G1,(1104,130), mask=image_ar_G1)
if get_user_input_r.a_findings_r==4:
    image2.paste(image_ar_G2,(1104,130), mask=image_ar_G2)
if get_user_input_r.a_findings_r==5:
    image2.paste(image_ar_G3,(1104,130), mask=image_ar_G3)
if get_user_input_r.a_findings_r==6:
    image2.paste(image_ar_G4,(1104,130), mask=image_ar_G4)
if get_user_input_r.a_findings_r==7:
    image2.paste(image_ar_G5,(1104,130), mask=image_ar_G5)

if get_user_input_r.t_findings_r==3:
    image2.paste(image_tr_G1, (1100, 493), mask=image_tr_G1)
if get_user_input_r.t_findings_r==4:
    image2.paste(image_tr_G2, (1100, 493), mask=image_tr_G2)
if get_user_input_r.t_findings_r==5:
    image2.paste(image_tr_G3, (1100, 493), mask=image_tr_G3)
if get_user_input_r.t_findings_r==6:
    image2.paste(image_tr_G4, (1100, 493), mask=image_tr_G4)
if get_user_input_r.t_findings_r==7:
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
col1.image(image2, use_column_width=True)

# Display probability of EPE for left and right lobe
col1.write('Probability of left EPE: ' + str(np.round_(prediction[:,1], decimals=2))[1:-1])
col1.write('Probability of right EPE: ' + str(np.round_(prediction_r[:,1], decimals=2))[1:-1])

# Display SHAP explanation
with st.beta_expander("See how the model explanations were determined"):
    st.write("""
             """)
    st.markdown("""Model explanations were calculated based on SHAP (SHapley Additive exPlanations) values,\
    originally developed by [Lundberg et al. (2006)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf).\
    This is an additive feature attribution method that satisfies all three properties of explanation models: local accuracy, missingness, and consistency.""")
    st.write("""
             """)
    st.markdown("""Accuracy: the output of the explanation model must match the output of the original model for a given prediction""")
    st.markdown("""Missingness: when a feature is missing, it should have no impact on the model""")
    st.markdown("""Consistency: a feature’s assigned attribution must match its degree of importance in the original model\
    (ie: if overall – % tissue involvement has the highest attribution, it must also have the highest feature importance\
    and the model must rely on this feature the most for a given prediction).""")
    st.write("""
             """)
    st.markdown("""SHAP allows us to understand why our model made a given prediction by simplifying our complex model into a\
    linear function of binary variables. This approach has previously been implemented to improve understanding\
    of [hypoxemia risk during anesthetic care](https://www.nature.com/articles/s41551-018-0304-0)""")
    st.markdown("""**Red bars**: Features that ***increase*** the risk of ssEPE""")
    st.markdown("""**Blue bars**: Features that ***decrease*** the risk of ssEPE""")
    st.markdown("""**Width of bars**: Importance of the feature. The wider it is, the greater impact it has on risk of ssEPE""")

with st.beta_expander("See how the model was developed"):
    st.write("""
             """)
    st.markdown("""A retrospective sample of 900 prostatic lobes (450 patients) from RP specimens at\
     Credit Valley Hospital, Mississauga, between 2010 and 2020, was used as the training cohort. Features\
     (ie: variables) included patient demographics, clinical, sonographic, and site-specific data from\
     transrectal ultrasound-guided prostate biopsy. The primary label (ie: outcome) of interest was the presence\
     of EPE in the ipsilateral lobe of the prostatectomy specimen. All pathology was reviewed by a dedicated\
    uro-pathologist. A previously developed [logistic regression model]\
    (https://bjui-journals.onlinelibrary.wiley.com/doi/full/10.1111/bju.13733), which has the highest performance out of\
     current predictive models for ssEPE, was used as the baseline model for comparison.""")
    st.write("""
             """)
    st.markdown("""Dimensionality reduction was performed by removing highly correlated features\
     (Pearson correlation > 0.8) and using a modified [Boruta](https://www.jstatsoft.org/article/view/v036i11/0)\
      algorithm. This method involves fitting all features to a random forest model and determining feature importance\
       by comparing the relevance of each feature to that of random noise. Given that our dataset contains both\
        categorical and numerical features, SHAP was specifically selected in lieu of impurity-based measures\
         to reduce bias towards high cardinality features.""")
    st.write("""
             """)
    st.markdown("""Using the final set of the most important and independent features, a ten-fold stratified\
     cross-validation method was performed to train a gradient-boosted model, optimize hyperparameters,\
      and for internal validation. In stratified cross-validation, the training cohort was randomly partitioned\
       into ten equal folds, with each fold containing the same percentage of positive ssEPE cases. Nine folds\
        were used for model training and hyperparameter tuning while the remaining fold made up the validation cohort.\
         This process was repeated ten times such that each fold served as the validation cohort once. Model\
          performance was determined based on the average performance across all ten validation cohorts to improve\
           generalizability of the models. All models were further externally validated using a testing cohort of\
            122 lobes (61 patients) from RP specimens at Mississauga Hospital, Mississauga, between 2016 and 2020.\
            Model performance was assessed by area under receiver-operating-characteristic curve (AUROC) and \
             precision-recall curve (AUPRC) analysis. Clinical utility was determined by [decision curve analysis]\
             (https://pubmed.ncbi.nlm.nih.gov/17099194/), in which the net benefit is plotted against various\
              threshold probabilities for three different treatment strategies: treat all, treat none, and treat only\
               those predicted to have ssEPE by our model.""")
    st.write("""
             """)
    st.markdown("""The incidence of ssEPE in the training and testing cohorts were 30.7 and 41.8%, respectively.\
     Our model outperformed the baseline model with a mean **AUROC of 0.81** vs 0.75 (p<0.01)\
      and **mean AUPRC of 0.69** vs 0.60, respectively. Similarly, our model performed favourably on the external\
       testing cohort with an **AUROC of 0.81** vs 0.76 (p=0.03) and **AUPRC of 0.78** vs 0.72. On decision curve\
        analysis, our ML model achieved a higher net benefit than the baseline model for threshold probabilities\
         between 0.15 to 0.65 (Figure 2). This translates to a **reduction in avoidable non-nerve-sparing radical\
          prostatectomies by 10 vs 4 per 100 patients at a threshold value of 0.2**.""")
    st.write("""
                 """)
    colA, colB, colC = st.beta_columns([1, 1, 2])
    ROC = PIL.Image.open('ROC.png')
    PRC = PIL.Image.open('PRC.png')
    DCA = PIL.Image.open('DCA (XGB vs Sayyid).png')
    colA.image(ROC, use_column_width=True)
    colB.image(PRC, use_column_width=True)
    colC.image(DCA, use_column_width=True)

# Display user input for left and right lobe
with st.beta_expander('User Input'):
    st.write(user_input)
    st.write(user_input_r)

st.text(" ")
st.text(" ")
st.text(" ")

# Display supporting institutions
#st.header('')
#image3 = PIL.Image.open('Supporting Institutions.png')
#st.image(image3, use_column_width=False)

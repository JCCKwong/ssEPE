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
st.write('Based on: upcoming paper...')

# Import Trained Model and Explainer
model = joblib.load('XGB ssEPE model V3.bz2')
explainer = joblib.load('XGB SHAP ssEPE.bz2')

# Calculate SHAP values
#features_list = list(features.columns)
#explainer = shap.TreeExplainer(model, features, model_output='probability')

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
        prostate_vol = st.number_input('Prostate volume (mL)', 0.0, 300.0)
    with st.sidebar.beta_expander('Side-specific variables (Left)',expanded=True):
        dre = st.selectbox('DRE positivity', options=list(CHOICES.keys()), format_func=format_func_yn)
        trus = st.selectbox('Hypoechoic nodule on TRUS', options=list(CHOICES.keys()), format_func=format_func_yn)
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
               #'Prostate volume': prostate_vol,
               #'DRE positive': dre,
               #'TRUS positive': trus,
               #'% site involvement': sum(i >= 3 for i in gleason_t) / 4,
               '% core involvement': p_core_total / t_core_total,
               'Worst Gleason': sort_g_p_inv['Gleason'].max(),
               #'% core involvement at worst Gleason': sort_g_p_inv.loc[sort_g_p_inv['Gleason'] == sort_g_p_inv['Gleason'].max(), '% core involvement'].iloc[0],
               #'Gleason at maximum % core': sort_g_p_inv.loc[sort_g_p_inv['% core involvement'] == sort_g_p_inv['% core involvement'].max(), 'Gleason'].iloc[0],
               'Maximum % core involvement': sort_g_p_inv['% core involvement'].max(),
               'Base findings': base_findings,
               #'Base % positive cores': base_p_core / base_t_core,
               'Base % core involvement': base_p_inv,
               #'Mid findings': mid_findings,
               #'Mid % positive cores': mid_p_core / mid_t_core,
               'Mid % core involvement': mid_p_inv,
               #'Apex findings': apex_findings,
               #'Apex % positive cores': apex_p_core / apex_t_core,
               #'Apex % core involvement': apex_p_inv,
               #'Transition zone findings': tz_findings,
               #'Transition zone % positive cores': tz_p_core / tz_t_core,
               'Transition zone % core involvement': tz_p_inv
               }

    # Save positive cores and cores taken at each site for annotated prostate diagram later
    get_user_input.b_p_core = base_p_core
    get_user_input.b_t_core = base_t_core
    get_user_input.m_p_core = mid_p_core
    get_user_input.m_t_core = mid_t_core
    get_user_input.a_p_core = apex_p_core
    get_user_input.a_t_core = apex_t_core
    get_user_input.t_p_core = tz_p_core
    get_user_input.t_t_core = tz_t_core

    pt_features = pd.DataFrame(pt_data, index = [0])
    return pt_features

# Store the left lobe user input into a variable
user_input = get_user_input()

def get_user_input_r():
    with st.sidebar.beta_expander('Side-specific variables (Right)', expanded=True):
        dre_r = st.selectbox('DRE positivity', options=list(CHOICES.keys()), format_func=format_func_yn, key=1)
        trus_r = st.selectbox('Hypoechoic nodule on TRUS', options=list(CHOICES.keys()), format_func=format_func_yn, key=1)
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
                 #'Prostate volume': user_input['Prostate volume'],
                 #'DRE positive': dre_r,
                 #'TRUS positive': trus_r,
                 #'% site involvement': sum(i >= 3 for i in gleason_t_r) / 4,
                 '% core involvement': p_core_total_r / t_core_total_r,
                 'Worst Gleason': sort_g_p_inv_r['Gleason'].max(),
                 #'% core involvement at worst Gleason': sort_g_p_inv_r.loc[sort_g_p_inv_r['Gleason'] == sort_g_p_inv_r['Gleason'].max(), '% core involvement'].iloc[0],
                 #'Gleason at maximum % core': sort_g_p_inv_r.loc[sort_g_p_inv_r['% core involvement'] == sort_g_p_inv_r['% core involvement'].max(), 'Gleason'].iloc[0],
                 'Maximum % core involvement': sort_g_p_inv_r['% core involvement'].max(),
                 'Base findings': base_findings_r,
                 #'Base % positive cores': base_p_core_r / base_t_core_r,
                 'Base % core involvement': base_p_inv_r,
                 #'Mid findings': mid_findings_r,
                 #'Mid % positive cores': mid_p_core_r / mid_t_core_r,
                 'Mid % core involvement': mid_p_inv_r,
                 #'Apex findings': apex_findings_r,
                 #'Apex % positive cores': apex_p_core_r / apex_t_core_r,
                 #'Apex % core involvement': apex_p_inv_r,
                 #'Transition zone findings': tz_findings_r,
                 #'Transition zone % positive cores': tz_p_core_r / tz_t_core_r,
                 'Transition zone % core involvement': tz_p_inv_r
                 }

    # Save positive cores and cores taken at each site for annotated prostate diagram later
    get_user_input_r.b_p_core_r = base_p_core_r
    get_user_input_r.b_t_core_r = base_t_core_r
    get_user_input_r.m_p_core_r = mid_p_core_r
    get_user_input_r.m_t_core_r = mid_t_core_r
    get_user_input_r.a_p_core_r = apex_p_core_r
    get_user_input_r.a_t_core_r = apex_t_core_r
    get_user_input_r.t_p_core_r = tz_p_core_r
    get_user_input_r.t_t_core_r = tz_t_core_r

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
                 #'Prostate volume',
                 #'DRE positive',
                 #'TRUS positive',
                 #'% site involvement',
                 '% core involvement',
                 'Worst Gleason',
                 #'% core involvement at worst Gleason',
                 #'Gleason at maximum % core',
                 'Maximum % core involvement',
                 'Base findings',
                 #'Base % positive cores',
                 'Base % core involvement',
                 #'Mid findings',
                 #'Mid % positive cores',
                 'Mid % core involvement',
                 #'Apex findings',
                 #'Apex % positive cores',
                 #'Apex % core involvement',
                 #'Transition zone findings',
                 #'Transition zone % positive cores',
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
                   #'Prostate volume',
                   #'DRE positive',
                   #'TRUS positive',
                   #'% site involvement',
                   '% core involvement',
                   'Worst Gleason',
                   #'% core involvement at worst Gleason',
                   #'Gleason at maximum % core',
                   'Maximum % core involvement',
                   'Base findings',
                   #'Base % positive cores',
                   'Base % core involvement',
                   #'Mid findings',
                   #'Mid % positive cores',
                   'Mid % core involvement',
                   #'Apex findings',
                   #'Apex % positive cores',
                   #'Apex % core involvement',
                   #'Transition zone findings',
                   #'Transition zone % positive cores',
                   'Transition zone % core involvement')
shap.force_plot(explainer.expected_value, shap_values_r, user_input_r, features_list_r, matplotlib=True, text_rotation=10)
col2.pyplot(bbox_inches='tight', dpi=600, pad_inches=0)
plt.clf()

# Show annotated prostate diagram under column 2
# Importing Image and ImageFont, ImageDraw module from PIL package
col1.header('Annotated Prostate')
col1.write('Automatically updates based on user entered values')
from PIL import ImageFont, ImageDraw, ImageOps

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

# Specify font size for annotated prostate diagram
font = ImageFont.truetype('arial.ttf', 50)

# Create text to overlay on annotated prostate diagram, auto-updates based on user inputted values
base_L = str(G_CHOICES[user_input['Base findings'][0]]) + '\n'\
         + 'Positive cores: ' + str(get_user_input.b_p_core) + '/' + str(get_user_input.b_t_core) + '\n'\
         + '% core inv: ' + str(user_input['Base % core involvement'][0])
mid_L = str(G_CHOICES[user_input['Mid findings'][0]]) + '\n'\
         + 'Positive cores: ' + str(get_user_input.m_p_core) + '/' + str(get_user_input.m_t_core) + '\n'\
         + '% core inv: ' + str(user_input['Mid % core involvement'][0])
apex_L = str(G_CHOICES[user_input['Apex findings'][0]]) + '\n'\
         + 'Positive cores: ' + str(get_user_input.a_p_core) + '/' + str(get_user_input.a_t_core) + '\n'\
         + '% core inv: ' + str(user_input['Apex % core involvement'][0])
tz_L = str(G_CHOICES[user_input['Transition zone findings'][0]]) + '\n'\
         + 'Positive cores: ' + str(get_user_input.t_p_core) + '/' + str(get_user_input.t_t_core) + '\n'\
         + '% core inv: ' + str(user_input['Transition zone % core involvement'][0])
base_R = str(G_CHOICES[user_input_r['Base findings'][0]]) + '\n'\
         + 'Positive cores: ' + str(get_user_input_r.b_p_core_r) + '/' + str(get_user_input_r.b_t_core_r) + '\n'\
         + '% core inv: ' + str(user_input_r['Base % core involvement'][0])
mid_R = str(G_CHOICES[user_input_r['Mid findings'][0]]) + '\n'\
         + 'Positive cores: ' + str(get_user_input_r.m_p_core_r) + '/' + str(get_user_input_r.m_t_core_r) + '\n'\
         + '% core inv: ' + str(user_input_r['Mid % core involvement'][0])
apex_R = str(G_CHOICES[user_input_r['Apex findings'][0]]) + '\n'\
         + 'Positive cores: ' + str(get_user_input_r.a_p_core_r) + '/' + str(get_user_input_r.a_t_core_r) + '\n'\
         + '% core inv: ' + str(user_input_r['Apex % core involvement'][0])
tz_R = str(G_CHOICES[user_input_r['Transition zone findings'][0]]) + '\n'\
         + 'Positive cores: ' + str(get_user_input_r.t_p_core_r) + '/' + str(get_user_input_r.t_t_core_r) + '\n'\
         + '% core inv: ' + str(user_input_r['Transition zone % core involvement'][0])

# Set conditions to show colour coded site images based on Gleason Grade Group for each site
draw = ImageDraw.Draw(image2)
if user_input['Base findings'][0]==3:
    image2.paste(image_bl_G1, (145, 958), mask=image_bl_G1)
if user_input['Base findings'][0]==4:
    image2.paste(image_bl_G2, (145, 958), mask=image_bl_G2)
if user_input['Base findings'][0]==5:
    image2.paste(image_bl_G3, (145, 958), mask=image_bl_G3)
if user_input['Base findings'][0]==6:
    image2.paste(image_bl_G4, (145, 958), mask=image_bl_G4)
if user_input['Base findings'][0]==7:
    image2.paste(image_bl_G5, (145, 958), mask=image_bl_G5)

if user_input['Mid findings'][0]==3:
    image2.paste(image_ml_G1, (145, 606), mask=image_ml_G1)
if user_input['Mid findings'][0]==4:
    image2.paste(image_ml_G2, (145, 606), mask=image_ml_G2)
if user_input['Mid findings'][0]==5:
    image2.paste(image_ml_G3, (145, 606), mask=image_ml_G3)
if user_input['Mid findings'][0]==6:
    image2.paste(image_ml_G4, (145, 606), mask=image_ml_G4)
if user_input['Mid findings'][0]==7:
    image2.paste(image_ml_G5, (145, 606), mask=image_ml_G5)

if user_input['Apex findings'][0]==3:
    image2.paste(image_al_G1, (145, 130), mask=image_al_G1)
if user_input['Apex findings'][0]==4:
    image2.paste(image_al_G2, (145, 130), mask=image_al_G2)
if user_input['Apex findings'][0]==5:
    image2.paste(image_al_G3, (145, 130), mask=image_al_G3)
if user_input['Apex findings'][0]==6:
    image2.paste(image_al_G4, (145, 130), mask=image_al_G4)
if user_input['Apex findings'][0]==7:
    image2.paste(image_al_G5, (145, 130), mask=image_al_G5)

if user_input['Transition zone findings'][0]==3:
    image2.paste(image_tl_G1, (665, 493), mask=image_tl_G1)
if user_input['Transition zone findings'][0]==4:
    image2.paste(image_tl_G2, (665, 493), mask=image_tl_G2)
if user_input['Transition zone findings'][0]==5:
    image2.paste(image_tl_G3, (665, 493), mask=image_tl_G3)
if user_input['Transition zone findings'][0]==6:
    image2.paste(image_tl_G4, (665, 493), mask=image_tl_G4)
if user_input['Transition zone findings'][0]==7:
    image2.paste(image_tl_G5, (665, 493), mask=image_tl_G5)

if user_input_r['Base findings'][0]==3:
    image2.paste(image_br_G1,(1104,958), mask=image_br_G1)
if user_input_r['Base findings'][0]==4:
    image2.paste(image_br_G2,(1104,958), mask=image_br_G2)
if user_input_r['Base findings'][0]==5:
    image2.paste(image_br_G3,(1104,958), mask=image_br_G3)
if user_input_r['Base findings'][0]==6:
    image2.paste(image_br_G4,(1104,958), mask=image_br_G4)
if user_input_r['Base findings'][0]==7:
    image2.paste(image_br_G5,(1104,958), mask=image_br_G5)

if user_input_r['Mid findings'][0]==3:
    image2.paste(image_mr_G1, (1542, 606), mask=image_mr_G1)
if user_input_r['Mid findings'][0]==4:
    image2.paste(image_mr_G2, (1542, 606), mask=image_mr_G2)
if user_input_r['Mid findings'][0]==5:
    image2.paste(image_mr_G3, (1542, 606), mask=image_mr_G3)
if user_input_r['Mid findings'][0]==6:
    image2.paste(image_mr_G4, (1542, 606), mask=image_mr_G4)
if user_input_r['Mid findings'][0]==7:
    image2.paste(image_mr_G5, (1542, 606), mask=image_mr_G5)

if user_input_r['Apex findings'][0]==3:
    image2.paste(image_ar_G1,(1104,130), mask=image_ar_G1)
if user_input_r['Apex findings'][0]==4:
    image2.paste(image_ar_G2,(1104,130), mask=image_ar_G2)
if user_input_r['Apex findings'][0]==5:
    image2.paste(image_ar_G3,(1104,130), mask=image_ar_G3)
if user_input_r['Apex findings'][0]==6:
    image2.paste(image_ar_G4,(1104,130), mask=image_ar_G4)
if user_input_r['Apex findings'][0]==7:
    image2.paste(image_ar_G5,(1104,130), mask=image_ar_G5)

if user_input_r['Transition zone findings'][0]==3:
    image2.paste(image_tr_G1, (1100, 493), mask=image_tr_G1)
if user_input_r['Transition zone findings'][0]==4:
    image2.paste(image_tr_G2, (1100, 493), mask=image_tr_G2)
if user_input_r['Transition zone findings'][0]==5:
    image2.paste(image_tr_G3, (1100, 493), mask=image_tr_G3)
if user_input_r['Transition zone findings'][0]==6:
    image2.paste(image_tr_G4, (1100, 493), mask=image_tr_G4)
if user_input_r['Transition zone findings'][0]==7:
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

# Display user input for left and right lobe
st.header('User Input')
st.write(user_input)
st.write(user_input_r)

# Display supporting institutions
st.header('')
image3 = PIL.Image.open('Supporting Institutions.png')
st.image(image3, use_column_width=False)
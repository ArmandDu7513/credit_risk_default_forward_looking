################################   Packages    ###################################
import streamlit as st
import pandas as pd
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
from functions_module import plot_series_brutes

################################   Main Config    ###################################

icon = Image.open("pictures/nexialog_icon.png")

st.set_page_config(page_title="PD Forward Looking", page_icon=icon)
# Streamit book properties


# create global variable : the dataframe to begin with
macro_base = pd.read_excel("data/variables_macroeconomiques.xlsx").set_index("Date")

file_path = ""
td = pd.read_excel(file_path + "data/default_rate_quarterly.xlsx")
macro = pd.read_excel(file_path + "data/variables_macroeconomiques.xlsx")
td = td.set_index("Date")
macro = macro.set_index("Date")
macro["td"] = td
macro.dropna(subset=["td"], how="all", inplace=True)

list_col_transformed_columns = (
    macro_base.add_suffix("_ann").columns.tolist()
    + macro_base.add_suffix("_tri").columns.tolist()
)

col1, col2, col3, col4 = st.columns(4)

with col4:
    st.image("pictures/nexialog.png")
with col1:
    st.image("pictures/paris1.jpg")

st.markdown("### Overview")
st.markdown("#### Présentation des séries brutes")
st.write(
    "Nous disposons de 5 séries macro-économiques et du taux de défaut. Les séries macro-économiques sont des séries brutes qui nécessitent de subir des transformations avant que l'on puisse les utiliser dans une quelconque modélisation. D'abord, on peut montrer à quoi ressemblent les séries macro. Elles sont exprimées en unités de mesures différentes (€, %...) /n Par ailleurs, on peut dès maintenant analyser les chocs et les tendances des séries"
)


plot_series_brutes(macro)

################################   Packages    ###################################
import streamlit as st
import extra_streamlit_components as stx

from PIL import Image
import pandas as pd
import numpy as np
from streamlit import session_state

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import kpss
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller
import plotly
import plotly.express as px
import time

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, kstest
from scipy.stats import kendalltau
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

from functions_module import list_str, plot_series_brutes, remove_season, lissage_moyenne_mobile, logit, import_and_series_transformation, import_and_series_transformation_and_graph, drop_trimestrielle_or_annuelle_if_wanted, highlight_variables, highlight_result, highlight_value, ljung_box, show_acf_pacf, AugmenteDickeyFuller_test, PhillipsPerron_test, kpss_test, concat_tests_and_sort, draw_conclusion_stationary, readable_conclusions, newey_west_tests_find_beta, compute_var_lagged, commpute_corr_matrix, choosing_seuil_corr_to_DR, commpute_corr_matrix_var_macro, find_high_corr_pairs


################################   Main Config    ###################################

icon = Image.open("pictures/nexialog_icon.png")

st.set_page_config(
    page_title="Projet Nexialog - Application PD Forward Looking",
    page_icon=icon
)



# Create a sidebar
st.sidebar.title("Projet Nexialog - Application PD Forward Looking")

# Create a dropdown menu for navigation
page_options = ["I - Transformation des séries temporelles","II - Variables Macroéconomiques : Stationnarité", "III - Taux de Défaut : Stationnarité", "IV - Corrélation des variables", "V - Prévoir DRt"]
selected_page = st.sidebar.selectbox("Chapitre...", page_options)

# create global variable : the dataframe to begin with
macro_base = pd.read_excel("data/variables_macroeconomiques.xlsx").set_index("Date")


######################################################################################
####################################   PAGE 1   ######################################
######################################################################################

if selected_page == "I - Transformation des séries temporelles":

    col1, col2 = st.columns(2)

    with col1:
        st.image('pictures/nexialog1.jpg')
    with col2:
        st.image('pictures/paris1.jpg')


    st.title("I - Transformation des séries temporelles")
    st.header("Présentation des séries brutes")
    st.write("Nous disposons de 5 séries macro-économiques et du taux de défaut. Les séries macro-économiques sont des séries brutes qui nécessitent de subir des transformations avant que l'on puisse les utiliser dans une quelconque modélisation. D'abord, on peut montrer à quoi ressemblent les séries macro. Elles sont exprimées en unités de mesures différentes (€, %...) /n Par ailleurs, on peut dès maintenant analyser les chocs et les tendances des séries")
    
    file_path = ""
    td = pd.read_excel(file_path+"data/default_rate_quarterly.xlsx")
    macro = pd.read_excel(file_path+"data/variables_macroeconomiques.xlsx")
    td = td.set_index("Date")
    macro = macro.set_index("Date")
    macro["td"]=td
    macro.dropna(subset=['td'], how='all', inplace=True)

    plot_series_brutes(macro)
    list_col_transformed_columns = macro_base.add_suffix('_ann').columns.tolist() + macro_base.add_suffix('_tri').columns.tolist()

    st.header("Choix des transformations")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.header('NaN')
        drop_na = st.radio("Voulez-vous retirer les valeurs manquantes en début et en fin de série liées au taux de défaut ?",
        ('Non', 'Oui'))

    with col2:
        st.header('Logit')
        logit_choix = st.radio("Voulez-vous appliquer une transformation sur le taux de défaut ?",
        ('Non', 'Oui'))

    with col3:
        st.header('Lissage')
        list_col_lissage = st.multiselect(
        "Quelles sont les variables que vous désirez rectifier en lissant les dernières observations (de sorte à retirer l'effet COVID, qui peut tromper la modélisation)?",
        list_col_transformed_columns, default = ['RGDP_ann', 'RGDP_tri'])
    st.session_state['list_col_lissage'] = list_col_lissage
    st.sidebar.write("Lissage sur la/les série/s :" , str(list(list_col_lissage)))
    with col4:
        st.header('Saisonnalité')
        list_col_saison = st.multiselect(
        'Quelles sont les variables que vous désirez rectifier en estimant puis en retirant la saisonnalité?',
        macro_base.columns, default = ['HICP'])
        #st.write('On retire la saisonnalité de :', list_col_saison)
    st.session_state['list_col_saison'] = list_col_saison
    st.sidebar.write("Sans saisonnalité pour la/les série/s :" , str(list(list_col_saison)))

    col1, col2 = st.columns([1,4])

    # Selection de la variable à montrer sur le graphique comparant la variable au taux de défaut &  Realisation du dataframe selon les transformations demandée
    with col1:
        variable_to_show = st.radio("On montre le graphique du taux de défaut comparé à la variable... : ", list_col_transformed_columns, key='radio_graph')
    with col2:
        macro = import_and_series_transformation_and_graph(drop_na, list_col_saison, list_col_lissage, logit_choix,  variable_to_show )
    
    st.header("Choix des types de variables")

    choix_des_variables = st.radio('Voulez-vous retirer un certain type de variables ? Vous pouvez retirer les transformations par variation trimestrielles pour ne garder que les variables qui représentent les variations annuelles. Inversement, vous pouvez choisir des garder les variables issues des variations annuelles, et retirer les autres. Nous conseillons de garder les deux types. Vous choisissez de garder... :', 
                       ('Les deux variations', 'Trimestrielles', 'Annuelles'))
    macro_kept = drop_trimestrielle_or_annuelle_if_wanted(macro, choix_des_variables)

    macro_kept_columns = ', '.join([str(elem) for i,elem in enumerate(macro_kept.columns)])
    st.session_state['series_kept'] = macro_kept_columns
    st.sidebar.write("Les séries utilisées pour les tests sont :" , macro_kept_columns)

    st.session_state.macro_kept = macro_kept
    show_df = st.checkbox('Voulez-vous voir le dataframe ?')
    if show_df:
        st.dataframe(st.session_state.macro_kept)

######################################################################################
####################################   PAGE 2   ######################################
######################################################################################

elif selected_page == "II - Variables Macroéconomiques : Stationnarité":

    st.title("II - Variables Macroéconomiques : Stationnarité")
    st.header("Présentation des Tests")
    st.write("La non-stationnarité a des conséquences fondamentales sur le plan économétrique. En présence d'une racine unitaire, les propriétés asymptotiques usuelles des estimateurs ne sont plus valables et il est nécessaire de développer une théorie asymptotique particulière.  \n Les variables non stationnaires sont exclus de l'étude.")
    st.write("ADF (Augmented Dickey Fuller)")
    st.write("On met en œuvre la régression ∆𝑋_𝑡 = 𝛾_0+𝜙_1 𝑋_(𝑡−1)+𝑢_𝑡. C'est un test unilatéral qui a pour hypothèses : \n H_0 : racine unitaire donc -> non-stationnarité. \n H_1 : pas de racine unitaire -> stationnarité")
    st.write("KPSS (Kwiatkowski-Phillips-Schmidt-Shin)")
    st.write("Le test de KPSS détermine si une série chronologique est stationnaire autour d’une tendance moyenne ou linéaire, ou non stationnaire en raison de la présence d’une racine unitaire.")
    st.write("Le test KPSS est basé sur une régression linéaire. Il décompose une série en trois parties : une tendance déterministe (𝛽𝑡), une marche aléatoire (𝛼) et une erreur stationnaire (𝜀_𝑡), avec l'équation de régression :")
    st.write("𝑋_𝑡 = ∅𝑋_(𝑡−1)+ 𝛼 + 𝛽𝑡 + 𝜀_𝑡.")
    st.write("Au final on teste : (𝐻0 :𝐿𝑎 𝑠é𝑟𝑖𝑒 𝑒𝑠𝑡 𝑠𝑡𝑎𝑡𝑖𝑜𝑛𝑛𝑎𝑖𝑟𝑒 𝐻1 :𝐿𝑎 𝑠é𝑟𝑖𝑒 𝑛' 𝑒𝑠𝑡 𝑝𝑎𝑠 𝑠𝑡𝑎𝑡𝑖𝑜𝑛𝑛𝑎𝑖𝑟𝑒)")
    st.write("Phillips Perron")
    st.write("SDFGSTGSH")
    st.write("H0 ... H1 ...")

    show_df = st.checkbox('On récupère le dataframe définit dans la partie I. Le voici :')
    macro = st.session_state.macro_kept
    macro_kept_statio = st.session_state.macro_kept

    if show_df:
        st.dataframe(macro)

    #st.header("Test sur les résidus : Test de Ljung-Box")
    #with st.expander("Ljung-Box"):
    #    st.write("Ce test vérifie si les séries sont des bruits blancs. Ce qui est une hypothèse plus forte que la stationnarité.")
    #    lj_b = ljung_box(macro, 'Yes')
    #    st.dataframe(lj_b.style.applymap(highlight_value, subset=['Résultat']))


    st.header('Analyse graphique : ACF et PACF')
    variable_choisie = st.selectbox(
        "Quelle est la variable que vous souhaitez analyser avec son ACF/PACF?",
    macro.drop('td', axis=1).columns)
    st.write('On observe les graphiques pour :', variable_choisie)
    show_acf_pacf(macro, variable_choisie)

    st.header('Tests de stationnarité : KPSS, Philips Perron, Augmented Dickey Fuller')

    st.write("Chaque test a une construction différente. On peut tester plusieurs hypothèses. Pour KPSS on peut choisir de tester la stationnarité, de chaque série, en supposant qu'elle admet une constante. On peut sinon admettre qu'elle contient un trend en plus d'une constante. On propose aussi de tester ces deux hypothèses simultanément /n Usuellement, on choisit de garder le moins d'attributs déterministes possible : Pour KPPS on prend souvent 'c', pour les autres tests on prend 'n' ('none').")

    st.header('Choix des hypothèses')

###############   KPSS   ####################

    st.write("Test KPSS : Choix des tests à effectuer.")
    agree_c_kpss = st.checkbox('Constante', value = True)
    agree_ct_kpss = st.checkbox('Constante et Trend', value=False)

    if agree_c_kpss:
        results_kpss = kpss_test(macro.drop('td', axis=1).dropna(), regression='c')
    if agree_ct_kpss:
        results_kpss = kpss_test(macro.drop('td', axis=1).dropna(), regression='ct')
    if agree_ct_kpss and agree_c_kpss:
        results_c = kpss_test(macro.drop('td', axis=1).dropna(), regression='c')
        results_ct = kpss_test(macro.drop('td', axis=1).dropna(), regression='ct')
        results_kpss = pd.concat([results_c, results_ct], ignore_index=True)

###############   PP & KPSS   ###############


    st.write("Tests ADF et Phillips-Perron : Choix des tests à effectuer.")
    agree_n_PP_ADF = st.checkbox('Aucun paramètre déterministe', value = True, key=1)
    agree_c_PP_ADF = st.checkbox('Constante', value = False, key=2)
    agree_ct_PP_ADF = st.checkbox('Constante et Trend', value=False, key=3)

    #st.write('Test Philips Perron')
    if agree_n_PP_ADF :
        results_pp = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='n')
    if agree_c_PP_ADF :
        results_pp = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='c')
    if agree_c_PP_ADF :
        results_pp = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='ct')
    if agree_n_PP_ADF and agree_c_PP_ADF:
        results_n = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='n')
        results_c = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='c')
        results_pp = pd.concat([results_n, results_c], ignore_index=True)
    if agree_n_PP_ADF and agree_ct_PP_ADF:
        results_n = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='n')
        results_ct = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='ct')
        results_pp = pd.concat([results_n, results_ct], ignore_index=True)
    if agree_c_PP_ADF and agree_ct_PP_ADF:
        results_c = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='c')
        results_ct = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='ct')
        results_pp = pd.concat([results_c, results_ct], ignore_index=True)
    if agree_n_PP_ADF and agree_c_PP_ADF and agree_ct_PP_ADF:
        results_n = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='n')
        results_c = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='c')
        results_ct = PhillipsPerron_test(macro.drop('td', axis=1).dropna(), regression='ct')
        results_pp = pd.concat([results_n, results_c, results_ct], ignore_index=True)

    #st.write('Test Augmented Dickey Fuller')
    
    if agree_n_PP_ADF :
        results_adf = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='n')
    if agree_c_PP_ADF :
        results_adf = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='c')
    if agree_c_PP_ADF :
        results_adf = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='ct')
    if agree_n_PP_ADF and agree_c_PP_ADF:
        results_n = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='n')
        results_c = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='c')
        results_adf = pd.concat([results_n, results_c], ignore_index=True)
    if agree_n_PP_ADF and agree_ct_PP_ADF:
        results_n = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='n')
        results_ct = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='ct')
        results_adf = pd.concat([results_n, results_ct], ignore_index=True)
    if agree_c_PP_ADF and agree_ct_PP_ADF:
        results_c = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='c')
        results_ct = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='ct')
        results_adf = pd.concat([results_c, results_ct], ignore_index=True)
    if agree_n_PP_ADF and agree_c_PP_ADF and agree_ct_PP_ADF:
        results_n = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='n')
        results_c = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='c')
        results_ct = AugmenteDickeyFuller_test(macro.drop('td', axis=1).dropna(), regression='ct')
        results_adf = pd.concat([results_n, results_c, results_ct], ignore_index=True)

    st.write('On veut regarder les résultats sur les séries...')
    agree_show_stationnaire = st.checkbox('Les séries stationnaires', value = True)
    agree_show_non_stationnaire = st.checkbox('Les séries non-stationnaires', value = True)
    if agree_show_stationnaire and not agree_show_non_stationnaire:
        results_kpss_show = results_kpss[results_kpss["Résultat"]=='Stationnaire']
        results_pp_show = results_pp[results_pp["Résultat"]=='Stationnaire']
        results_adf_show = results_adf[results_adf["Résultat"]=='Stationnaire']
    if not agree_show_stationnaire and agree_show_non_stationnaire:
        results_kpss_show = results_kpss[results_kpss["Résultat"]!='Stationnaire']
        results_pp_show = results_pp[results_pp["Résultat"]!='Stationnaire']
        results_adf_show = results_adf[results_adf["Résultat"]!='Stationnaire']
    if agree_show_stationnaire and agree_show_non_stationnaire:
        results_kpss_show = results_kpss
        results_pp_show = results_pp
        results_adf_show = results_adf

    styled_results_kpss = results_kpss_show.style.applymap(highlight_result, subset=['Résultat'])
    styled_results_pp = results_pp_show.style.applymap(highlight_result, subset=['Résultat'])
    styled_results_adf = results_adf_show.style.applymap(highlight_result, subset=['Résultat'])

    st.write('Test KPSS')
    st.write(styled_results_kpss)
    st.write('Test Philips Perron')
    st.write(styled_results_pp)
    st.write('Test Augmented Dickey Fuller')
    st.write(styled_results_adf)


    st.header('Conclusion')
    st.write('On va concaténer les résultats des 3 tests afin de compter le nombre de tests qui concluent en faveur de la stationnarité par rapport à ceux qui concluent en défaveur de la stationnarité. On exprique ce résultat en Pourcentage : plus il est élevé, plus il est probable que la série soit stationnaire.')

    resultats_des_tests = concat_tests_and_sort(results_kpss, results_pp, results_adf)
    nb_variables_studies = len(resultats_des_tests['Variable'].unique())
    conclusions = draw_conclusion_stationary(resultats_des_tests, nb_variables_studies)
    conclusions_readable = readable_conclusions(conclusions)
    st.dataframe(conclusions_readable.style.applymap(highlight_result, subset=['Décision Finale']))

    st.write("Quand plus de 50% des tests concluent que la série est stationnaire, on penche en faveur de la stationnarité. Sinon, on décide de mettre cette variable, elle ne sera pas utilisée dans les prochaines étapes, car elle est inutilisable pour l'étape de la modélisation.")

    list_var_stationary = conclusions_readable[conclusions_readable['Décision Finale']=='Stationnaire']['Variable'].unique().tolist()

    show_df_out = st.checkbox('Le dataframe en output selon les résultats des tests est le suivant:')
    
    if show_df_out:
        st.dataframe(macro[list_var_stationary+ ['td']])
    
    st.session_state.macro_stationary = macro[list_var_stationary+ ['td']]

    st.write('Maintenant que nous avons déterminé quelles sont les variables macroéconomiques qui sont stationnaires, on doit étudier la stationnarité du taux de défaut.')

    list_var_stationary_show = ', '.join([str(elem) for i,elem in enumerate(macro[list_var_stationary+ ['td']])])
    st.session_state.macro_kept_statio = macro_kept_statio[list_var_stationary+ ['td']]

    st.session_state['list_var_stationary_show'] = list_var_stationary_show
    st.sidebar.write("Les séries testées pour les corrélations sont " , list_var_stationary_show)

######################################################################################
####################################   PAGE 3   ######################################
######################################################################################
elif selected_page == "III - Taux de Défaut : Stationnarité":
 
    st.title("III - Taux de Défaut : Stationnarité")

    macro = st.session_state.macro_kept
    td_serie = pd.DataFrame(macro['td']).dropna()

    #st.header("Test sur les résidus : Test de Ljung-Box")
    #with st.expander("Ljung-Box"):
    #    lj_b = ljung_box(td_serie, 'No')
    #    st.dataframe(lj_b.style.applymap(highlight_value, subset=['Résultat']))

    st.header('Analyse graphique : ACF et PACF')
    with st.expander("Graphiques"):
        show_acf_pacf(td_serie, 'td')

    st.header('Tests de stationnarité : KPSS, Philips Perron, Augmented Dickey Fuller')
    
    results_KPSS_c = kpss_test(td_serie, regression='c')
    results_PP_c = PhillipsPerron_test(td_serie,  regression='c')
    results_ADF_c = AugmenteDickeyFuller_test(td_serie, regression='c')
    results_DR = pd.concat([results_KPSS_c, results_PP_c, results_ADF_c], ignore_index=True)
    st.dataframe(results_DR.style.applymap(highlight_result, subset=['Résultat']))
    st.write("On voit rapidement entre les graphiques et les tests, que le taux de défaut n'est absolument pas stationnaire. On va dont passer par la différenciation partielle.")

    st.header('Différenciation partielle du Taux de défaut')
    st.write("Ici on a compris que le taux de défaut n'était pas stationnaire. Automatiquement, on pense à appliquer la différence première et regarder à nouveau s'il est stationnaire. Cependant différencier le taux de défaut est une mauvaise pratique. On préfère alors regarder pour quel coefficient Beta, la série de Différence Partielle est stationnaire. /n Ici on teste l'équation : DRt - Beta * DRt-1. Et on en retire les Betas pour lesquels l'équation est stationnaire")

    
    kpss_beta = newey_west_tests_find_beta(macro, 'KPSS')

    with st.expander("Intervalles selon les autres tests de stationnarité"):
        pp_beta = newey_west_tests_find_beta(macro, 'PP')
        adf_beta = newey_west_tests_find_beta(macro, 'ADF')

    st.header("Choix de l'intervalle du Beta")
    st.write("Ainsi, on peut choisir un intervalle pour le coefficient. On estimera plustard les modèles pour prédire le taux de défaut, et ceux-ci devront vérifier la condition énoncé par l'intervalle ici déclaré.")

    values_adf = np.array(adf_beta[adf_beta['crit'] > adf_beta['stat']]['betax'])
    values_kpss = np.array(kpss_beta[kpss_beta['crit'] > kpss_beta['stat']]['betax'])
    values_pp = np.array(pp_beta[pp_beta['crit'] > pp_beta['stat']]['betax'])

    # Inner join sur les deux arrays : car on veut garder les valeurs de beta pour lesquellesles deux tests expriment la stationnarité
    
    st.write("Choix des tests à partir desquels on choisit l'intervalle du Beta")
    agree_intervalle_kpss = st.checkbox('Intervalle KPSS', value = True, key=1)
    agree_intervalle_pp = st.checkbox('Intervalle PP', value = False, key=2)
    agree_intervalle_adf = st.checkbox('Intervalle ADF', value=False, key=3)
    
    if agree_intervalle_kpss:
        values_available_for_beta = values_kpss

    if agree_intervalle_pp:
        values_available_for_beta = values_pp

    if agree_intervalle_adf:
        values_available_for_beta = values_adf

    if agree_intervalle_kpss and agree_intervalle_pp:
        values_available_for_beta = np.intersect1d(values_kpss, values_pp)
    
    if agree_intervalle_kpss and agree_intervalle_adf:
        values_available_for_beta = np.intersect1d(values_kpss, values_adf)
    
    if agree_intervalle_pp and agree_intervalle_adf:
        values_available_for_beta = np.intersect1d(values_pp, values_adf)
    if agree_intervalle_pp and agree_intervalle_adf and agree_intervalle_kpss:
        values = np.intersect1d(values_pp, values_adf)
        values_available_for_beta = np.intersect1d(values, values_kpss)

    if not len(values_available_for_beta) == 0:
        st.text(values_available_for_beta.T)
    else:
        st.write('Aucun Beta disponible, les tests sont trop restrictifs')



######################################################################################
####################################   PAGE 4   ######################################
######################################################################################

elif selected_page == "IV - Corrélation des variables":
 
    st.title("IV - Corrélation des variables")
    macro_var_chosen = st.session_state.macro_kept_statio

    st.title("Calcul des variables explicatives retardées")
    st.write("On a maintenant seulement les variables stationnaires. On peut alors choisir de prendre les 'lags'pour créer des retards des variables. On laisse le choix à l'utilisateur du nombre de retards qu'on accorde aux variables. On peut mettre un nombre de retards différent pour les variables avec variation trimestrielle et avec variation annuelle. Il faut faire attention car plus on ajoute de retards, plus on crée des variables manquantes en début de période. Les variables avec variation annuelle ont déjà plusieurs valeurs manquantes en début de période. C'est pourquoi on conseille de bien garder un nombre de retards plus faible pour les varibles en variations annuelles.")

    default_value_ann = 2
    default_value_tri = 5

    number_lag_ann = st.number_input('Choisir un nombre de retards pour les variables annuelles',min_value = 0, max_value = 8, value = default_value_ann)
    st.write('Nombre de retards maximum :', number_lag_ann)

    number_lag_tri = st.number_input('Choisir un nombre de retards pour les variables trimestrielles',min_value = 0, max_value = 8, value = default_value_tri)
    st.write('Nombre de retards maximum :', number_lag_tri)
    number_lag_td=1
    macro_lagged = compute_var_lagged(st.session_state.macro_kept_statio, number_lag_ann, number_lag_tri, number_lag_td)
    macro_lagged_for_model = macro_lagged.copy()
    show_datafr = st.checkbox('Voir le dataframe, avec les variables retardées.')
    if show_datafr:
        st.dataframe(macro_lagged)

    show_list = st.checkbox('Voir la liste des variables, avec retard.')
    if show_list:
        st.write(macro_lagged.columns)

    st.header("Corrélation : Taux de défaut versus Variables macro")

    corr_df = commpute_corr_matrix(macro_lagged)
    # On se focalise sur les correlations concernant le taux de défaut
    df_var_corr_to_DR = pd.DataFrame(corr_df['td']).rename(columns={'td': 'Kendall Correlations'}).sort_values(by=['Kendall Correlations']).drop('td',axis=0)
    #peu importe le sens des signes, on cherche surtout à avoir les variables qui sont corrélées avec le taux de défaut : donc on prend la valeur absolue
    df_var_corr_to_DR['Kendall Correlations'] = df_var_corr_to_DR['Kendall Correlations'].abs()
    df_var_corr_to_DR = df_var_corr_to_DR.sort_values(by=['Kendall Correlations'], ascending=False)
    
    show_datafr_corr_td = st.checkbox('Voir les corrélations des variables (corrélations absolues) avec le taux de défaut.')
    if show_datafr_corr_td:
        st.dataframe(df_var_corr_to_DR)

    default_value_threshold = 0.15
    number_threshold = st.number_input('Choisir un seuil minimal de corrélation pour utilser la variable',min_value = 0.00, max_value = 1.00, value = default_value_threshold)
    st.write('Seuil =', number_threshold)

    list_var_in_model = choosing_seuil_corr_to_DR(df_var_corr_to_DR, number_threshold)
    
    show_list_var_corr = st.checkbox("Voir la liste des variables macroéconomiques qu'on souhaite garder car elles sont assez corrélées avec le taux de défaut")
    if show_list_var_corr:
        st.write(sorted(list_var_in_model))

    ######################################################
    st.header("Corrélation : Les variables macro entre elles")
    st.write(
            "A travers l'étude des corrélations, on peut anticiper une trop forte corrélation entre des variables. Surtout lorsqu'il s'agit de la même variable, avec un retard différent. On propose de retirer manuellement des variables")
    st.write(
            "D'abord, on montre à gauche les corrélations de toutes les variables. Puis sur la partie de droite on laisse la possibilité de retirer des varaibles, pour que le seuil de corrélation indiqué ne soit pas dépassé.")
            
    macro_lagged_seuil = macro_lagged.copy()
    macro_lagged_for_corr = macro_lagged.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Corrélation : Point de départ')
        st.write(" ")
        corr_df_macro = commpute_corr_matrix_var_macro(macro_lagged, sorted(list_var_in_model))
        show_list_var_corr_df = st.checkbox("Voir le dataframe avec toutes les varaibles, que les corrélations soient élevées ou non.")
        if show_list_var_corr_df:
            st.dataframe(corr_df_macro)
        fig, ax = plt.subplots()
        sns.heatmap(corr_df_macro.astype(float), cmap='coolwarm', center=0, annot=False)
        st.pyplot(fig)


    with col2:
        st.subheader('Choix du seuil')
        # on définit le seuil de corrélation.
        default_seuil = 0.6
        number_seuil = st.number_input('',min_value = 0.00, max_value = 1.00, value = default_seuil)

        first_elements_variables = [t[0] for t in find_high_corr_pairs(corr_df_macro, number_seuil)]
        all_elements_variables = list(set(first_elements_variables))

        macro_lagged_for_corr = macro_lagged_for_corr[list_var_in_model].drop(all_elements_variables, axis=1)

        corr_df_for_corr = commpute_corr_matrix_var_macro(macro_lagged_for_corr, sorted(macro_lagged_for_corr.columns))
    
        fig, ax = plt.subplots()
        sns.heatmap(corr_df_for_corr.astype(float), cmap='coolwarm', center=0, annot=False)
        st.pyplot(fig)
    
    show_df_corr_all = st.checkbox('Voulez-vous voir le dataframe prêt pour la modélisation?')
    if show_df_corr_all:
            st.dataframe(macro_lagged[corr_df_for_corr.columns])

    st.session_state.macro_model = macro_lagged_for_model[list(corr_df_for_corr.columns) + ['td']]

######################################################################################
####################################   PAGE 5   ######################################
######################################################################################

elif selected_page == "V - Prévoir DRt":
 
    st.title("V - Prévoir DRt")
    macro_model = st.session_state.macro_model

    df_lags = macro_model.copy()

    df_lags = df_lags.dropna()

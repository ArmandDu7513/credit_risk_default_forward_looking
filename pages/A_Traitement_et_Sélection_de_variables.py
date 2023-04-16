################################   Packages    ###################################
import streamlit as st
import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns



import warnings

warnings.filterwarnings("ignore")

from functions_module import (
    list_str,
    plot_series_brutes,
    remove_season,
    lissage_moyenne_mobile,
    logit,
    import_and_series_transformation,
    import_and_series_transformation_and_graph,
    drop_trimestrielle_or_annuelle_if_wanted,
    highlight_variables,
    highlight_result,
    highlight_value,
    ljung_box,
    show_acf_pacf,
    AugmenteDickeyFuller_test,
    PhillipsPerron_test,
    kpss_test,
    concat_tests_and_sort,
    draw_conclusion_stationary,
    readable_conclusions,
    newey_west_tests_find_beta,
    compute_var_lagged,
    commpute_corr_matrix,
    choosing_seuil_corr_to_DR,
    commpute_corr_matrix_var_macro,
    find_high_corr_pairs,
)


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
st.markdown("### Traitement des variables macroéconomiques et du défaut")

tab1, tab2, tab3 = st.tabs(
    [
        "Transformation des séries temporelles",
        "Stationnarité",
        "Corrélations",
    ]
)

with tab1:
    st.markdown(
        "#### Pré-traitement des variables: Lissage, saisonnalité, valeurs manquantes"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Taux de défaut")
        col1_, col2_ = st.columns(2)
        with col1_:
            st.markdown("###### Valeurs manquantes")
            drop_na = st.radio(
                "Voulez-vous retirer les valeurs manquantes en début et en fin de série liées au taux de défaut ?",
                ("Non", "Oui"),
            )

        with col2_:
            st.markdown("###### Transformation Logit")
            logit_choix = st.radio(
                "Voulez-vous appliquer une transformation sur le taux de défaut ?",
                ("Non", "Oui"),
            )

    with col2:
        st.markdown("##### Variables macroéconomiques")
        col1_bis, col2_bis = st.columns(2)
        with col1_bis:
            st.markdown("###### Lissage")
            list_col_lissage = st.multiselect(
                "Quelles sont les variables que vous désirez rectifier en lissant les dernières observations (de sorte à retirer l'effet COVID, qui peut tromper la modélisation)?",
                list_col_transformed_columns,
                default=["RGDP_ann", "RGDP_tri", "RREP_tri", "RREP_ann"],
            )
            st.session_state["list_col_lissage"] = list_col_lissage
        with col2_bis:
            st.markdown("###### Extraction de la saisonnalité")
            list_col_saison = st.multiselect(
                "Quelles sont les variables que vous désirez rectifier en estimant puis en retirant la saisonnalité?",
                macro_base.columns,
                default=["HICP"],
            )
            # st.write('On retire la saisonnalité de :', list_col_saison)
        st.session_state["list_col_saison"] = list_col_saison
        # st.sidebar.write("Sans saisonnalité pour la/les série/s :" , str(list(list_col_saison)))

    st.markdown("# ")
    with st.expander(
        "Aperçu des évolutions des variables macroéconomiques avec le taux de défaut",
        expanded=True,
    ):
        col1, col2 = st.columns([1, 4])
        st.markdown("# ")
        st.markdown("# ")

        # Selection de la variable à montrer sur le graphique comparant la variable au taux de défaut &  Realisation du dataframe selon les transformations demandée
        with col1:
            st.markdown("# ")

            variable_to_show = st.radio(
                "*Sélectionnez la variable à comparer*",
                list_col_transformed_columns,
                key="radio_graph",
            )
        with col2:
            fig_pd_plot, macro = import_and_series_transformation_and_graph(
                drop_na,
                list_col_saison,
                list_col_lissage,
                logit_choix,
                variable_to_show,
            )
            st.plotly_chart(fig_pd_plot)

    st.markdown("# ")
    st.markdown(
        "**Souhaitez-vous conserver uniquement les features trimestriels/annuels?**"
    )
    choix_des_variables = st.radio(
        "*Vous pouvez retirer les transformations par variation trimestrielle pour ne garder que les variables qui représentent les variations annuelles. Inversement, vous pouvez choisir des garder les variables issues des variations annuelles, et retirer les autres. Nous conseillons de garder les deux types.*",
        ("Les deux variations", "Trimestrielles", "Annuelles"),
    )
    macro_kept = drop_trimestrielle_or_annuelle_if_wanted(macro, choix_des_variables)

    macro_kept_columns = ", ".join(
        [str(elem) for i, elem in enumerate(macro_kept.columns)]
    )
    st.session_state["series_kept"] = macro_kept_columns
    # st.sidebar.write("Les séries utilisées pour les tests sont :" , macro_kept_columns)

    st.session_state.macro_kept = macro_kept
    with st.expander("Aperçu des variables sélectionnées", expanded=True):
        st.dataframe(st.session_state.macro_kept.head(5))


with tab2:
    st.markdown("#### Stationnarité ")
    macro = st.session_state.macro_kept
    macro_kept_statio = st.session_state.macro_kept

    with st.expander("Présentation des Tests", expanded=False):
        st.write(
            "La non-stationnarité a des conséquences économétriques importantes. Les variables non stationnaires sont exclus de l'étude."
        )
        st.write("**ADF (Augmented Dickey Fuller)**")
        st.write("Teste la présence d'une racine unitaire :")
        st.latex(
            r"""
        \begin{cases}
        H_0: & \text{non-stationnarité} \\
        H_1: & \text{stationnarité}
        \end{cases}
    """
        )
        st.write("**KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**")
        st.write("Teste si une série est stationnaire autour d’une tendance :")
        st.latex(
            r"""
        \begin{cases}
        H_0: & \text{stationnarité} \\
        H_1: & \text{non-stationnarité}
        \end{cases}
    """
        )
        st.write("**Phillips Perron**")
        st.write(
            "Similaire à ADF, mais tient compte des autocorrélations et hétéroscédasticités"
        )
        st.latex(
            r"""
        \begin{cases}
        H_0: & \text{non-stationnarité} \\
        H_1: & \text{stationnarité}
        \end{cases}
    """
        )
    # st.header("Test sur les résidus : Test de Ljung-Box")
    # with st.expander("Ljung-Box"):
    #    st.write("Ce test vérifie si les séries sont des bruits blancs. Ce qui est une hypothèse plus forte que la stationnarité.")
    #    lj_b = ljung_box(macro, 'Yes')
    #    st.dataframe(lj_b.style.applymap(highlight_value, subset=['Résultat']))

    with st.expander("Analyse graphique : ACF et PACF"):
        variable_choisie = st.selectbox(
            "Quelle est la variable que vous souhaitez analyser avec son ACF/PACF?",
            macro.drop("td", axis=1).columns,
        )
        show_acf_pacf(macro, variable_choisie)

    st.markdown("##### Tests de stationnarité sur les variables macroéconomiques")

    st.write(
        "Les tests de stationnarité (ADF, KPSS, Phillips Perron) ont des constructions différentes et permettent de tester diverses hypothèses. Pour KPSS, on teste souvent la stationnarité avec une constante ('c'). Pour les autres tests, on utilise généralement 'n' (aucun attribut déterministe)."
    )

    st.markdown("###### Choix des hypothèses")

    ###############   KPSS   ####################
    tab1_test, tab2_test = st.tabs(["KPSS", "ADF & Phillips-Perron"])
    with tab1_test:
        agree_c_kpss = st.checkbox("Constante", value=True)
        agree_ct_kpss = st.checkbox("Constante et Trend", value=False)

        if agree_c_kpss:
            results_kpss = kpss_test(macro.drop("td", axis=1).dropna(), regression="c")
        if agree_ct_kpss:
            results_kpss = kpss_test(macro.drop("td", axis=1).dropna(), regression="ct")
        if agree_ct_kpss and agree_c_kpss:
            results_c = kpss_test(macro.drop("td", axis=1).dropna(), regression="c")
            results_ct = kpss_test(macro.drop("td", axis=1).dropna(), regression="ct")
            results_kpss = pd.concat([results_c, results_ct], ignore_index=True)

    ###############   PP & KPSS   ###############
    with tab2_test:
        agree_n_PP_ADF = st.checkbox("Aucun paramètre déterministe", value=True, key=1)
        agree_c_PP_ADF = st.checkbox("Constante", value=False, key=2)
        agree_ct_PP_ADF = st.checkbox("Constante et Trend", value=False, key=3)

        # st.write('Test Philips Perron')
        if agree_n_PP_ADF:
            results_pp = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="n"
            )
        if agree_c_PP_ADF:
            results_pp = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="c"
            )
        if agree_c_PP_ADF:
            results_pp = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="ct"
            )
        if agree_n_PP_ADF and agree_c_PP_ADF:
            results_n = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="n"
            )
            results_c = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="c"
            )
            results_pp = pd.concat([results_n, results_c], ignore_index=True)
        if agree_n_PP_ADF and agree_ct_PP_ADF:
            results_n = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="n"
            )
            results_ct = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="ct"
            )
            results_pp = pd.concat([results_n, results_ct], ignore_index=True)
        if agree_c_PP_ADF and agree_ct_PP_ADF:
            results_c = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="c"
            )
            results_ct = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="ct"
            )
            results_pp = pd.concat([results_c, results_ct], ignore_index=True)
        if agree_n_PP_ADF and agree_c_PP_ADF and agree_ct_PP_ADF:
            results_n = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="n"
            )
            results_c = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="c"
            )
            results_ct = PhillipsPerron_test(
                macro.drop("td", axis=1).dropna(), regression="ct"
            )
            results_pp = pd.concat(
                [results_n, results_c, results_ct], ignore_index=True
            )

        # st.write('Test Augmented Dickey Fuller')

        if agree_n_PP_ADF:
            results_adf = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="n"
            )
        if agree_c_PP_ADF:
            results_adf = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="c"
            )
        if agree_c_PP_ADF:
            results_adf = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="ct"
            )
        if agree_n_PP_ADF and agree_c_PP_ADF:
            results_n = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="n"
            )
            results_c = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="c"
            )
            results_adf = pd.concat([results_n, results_c], ignore_index=True)
        if agree_n_PP_ADF and agree_ct_PP_ADF:
            results_n = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="n"
            )
            results_ct = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="ct"
            )
            results_adf = pd.concat([results_n, results_ct], ignore_index=True)
        if agree_c_PP_ADF and agree_ct_PP_ADF:
            results_c = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="c"
            )
            results_ct = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="ct"
            )
            results_adf = pd.concat([results_c, results_ct], ignore_index=True)
        if agree_n_PP_ADF and agree_c_PP_ADF and agree_ct_PP_ADF:
            results_n = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="n"
            )
            results_c = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="c"
            )
            results_ct = AugmenteDickeyFuller_test(
                macro.drop("td", axis=1).dropna(), regression="ct"
            )
            results_adf = pd.concat(
                [results_n, results_c, results_ct], ignore_index=True
            )
    st.markdown("###### Résultats des tests")
    with st.expander("Résultats détaillés pour chaque test"):
        st.write("On veut regarder les résultats sur les séries...")
        agree_show_stationnaire = st.checkbox("Les séries stationnaires", value=True)
        agree_show_non_stationnaire = st.checkbox(
            "Les séries non-stationnaires", value=True
        )
        if agree_show_stationnaire and not agree_show_non_stationnaire:
            results_kpss_show = results_kpss[results_kpss["Résultat"] == "Stationnaire"]
            results_pp_show = results_pp[results_pp["Résultat"] == "Stationnaire"]
            results_adf_show = results_adf[results_adf["Résultat"] == "Stationnaire"]
        if not agree_show_stationnaire and agree_show_non_stationnaire:
            results_kpss_show = results_kpss[results_kpss["Résultat"] != "Stationnaire"]
            results_pp_show = results_pp[results_pp["Résultat"] != "Stationnaire"]
            results_adf_show = results_adf[results_adf["Résultat"] != "Stationnaire"]
        if agree_show_stationnaire and agree_show_non_stationnaire:
            results_kpss_show = results_kpss
            results_pp_show = results_pp
            results_adf_show = results_adf

        styled_results_kpss = results_kpss_show.style.applymap(
            highlight_result, subset=["Résultat"]
        )
        styled_results_pp = results_pp_show.style.applymap(
            highlight_result, subset=["Résultat"]
        )
        styled_results_adf = results_adf_show.style.applymap(
            highlight_result, subset=["Résultat"]
        )

        st.markdown("**Test KPSS**")
        st.write(styled_results_kpss)
        st.markdown("**Test Philips Perron**")
        st.write(styled_results_pp)
        st.markdown("**Test Augmented Dickey Fuller**")
        st.write(styled_results_adf)

    with st.expander("Conclusion", expanded=True):
        st.write(
            "Nous allons combiner les résultats des 3 tests (ADF, KPSS, Phillips Perron) pour évaluer la stationnarité de chaque série. En comptant le nombre de tests en faveur de la stationnarité par rapport à ceux en défaveur, nous obtenons un pourcentage. Un pourcentage élevé indique une probabilité plus forte que la série soit stationnaire."
        )

        resultats_des_tests = concat_tests_and_sort(
            results_kpss, results_pp, results_adf
        )
        nb_variables_studies = len(resultats_des_tests["Variable"].unique())
        conclusions = draw_conclusion_stationary(
            resultats_des_tests, nb_variables_studies
        )
        conclusions_readable = readable_conclusions(conclusions)
        st.dataframe(
            conclusions_readable.style.applymap(
                highlight_result, subset=["Décision Finale"]
            )
        )

        st.caption(
            "Si plus de 50% des tests indiquent la stationnarité, nous considérons la série comme stationnaire. Sinon, nous excluons cette variable des étapes ultérieures de modélisation car elle n'est pas adaptée."
        )

        list_var_stationary = (
            conclusions_readable[
                conclusions_readable["Décision Finale"] == "Stationnaire"
            ]["Variable"]
            .unique()
            .tolist()
        )

    st.session_state.macro_stationary = macro[list_var_stationary + ["td"]]

    list_var_stationary_show = ", ".join(
        [str(elem) for i, elem in enumerate(macro[list_var_stationary + ["td"]])]
    )
    st.session_state.macro_kept_statio = macro_kept_statio[list_var_stationary + ["td"]]

    st.session_state["list_var_stationary_show"] = list_var_stationary_show
    # st.sidebar.write("Les séries testées pour les corrélations sont " , list_var_stationary_show)

    st.markdown("##### Tests de stationnarité du taux de défaut")

    macro = st.session_state.macro_kept
    td_serie = pd.DataFrame(macro["td"]).dropna()

    # st.header("Test sur les résidus : Test de Ljung-Box")
    # with st.expander("Ljung-Box"):
    #    lj_b = ljung_box(td_serie, 'No')
    #    st.dataframe(lj_b.style.applymap(highlight_value, subset=['Résultat']))

    with st.expander("Analyse graphique : ACF et PACF"):
        show_acf_pacf(td_serie, "td")

    with st.expander(
        "Tests de stationnarité : KPSS, Philips Perron, Augmented Dickey Fuller"
    ):
        results_KPSS_c = kpss_test(td_serie, regression="c")
        results_PP_c = PhillipsPerron_test(td_serie, regression="c")
        results_ADF_c = AugmenteDickeyFuller_test(td_serie, regression="c")
        results_DR = pd.concat(
            [results_KPSS_c, results_PP_c, results_ADF_c], ignore_index=True
        )
        st.dataframe(results_DR.style.applymap(highlight_result, subset=["Résultat"]))
    st.write(
        "On voit rapidement entre les graphiques et les tests, que le taux de défaut n'est absolument pas stationnaire. On va dont passer par la différenciation partielle."
    )

    st.markdown("###### Etude de stationnarité sur différenciation partielle: Beta")
    st.write(
        "Le taux de défaut n'étant pas stationnaire, on évite de différencier directement. À la place, on cherche le coefficient Beta qui rend la série de Différence Partielle stationnaire. On teste l'équation pour identifier les Beta: "
    )
    st.latex(r"""DRt - Beta * DRt-1""")
    st.markdown("""#""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""**Graphique des Beta**""")

        kpss_beta = newey_west_tests_find_beta(macro, "KPSS")

        with st.expander("Intervalles selon les autres tests de stationnarité"):
            pp_beta = newey_west_tests_find_beta(macro, "PP")
            adf_beta = newey_west_tests_find_beta(macro, "ADF")
    with col2:
        st.markdown("**Choix de l'intervalle du Beta**")

        values_adf = np.array(adf_beta[adf_beta["crit"] > adf_beta["stat"]]["betax"])
        values_kpss = np.array(
            kpss_beta[kpss_beta["crit"] > kpss_beta["stat"]]["betax"]
        )
        values_pp = np.array(pp_beta[pp_beta["crit"] > pp_beta["stat"]]["betax"])

        # Inner join sur les deux arrays : car on veut garder les valeurs de beta pour lesquellesles deux tests expriment la stationnarité

        st.write("Choix des tests à partir desquels on choisit l'intervalle du Beta")
        agree_intervalle_kpss = st.checkbox("Intervalle KPSS", value=True, key=4)
        agree_intervalle_pp = st.checkbox("Intervalle PP", value=False, key=5)
        agree_intervalle_adf = st.checkbox("Intervalle ADF", value=False, key=6)

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
            st.write("Aucun Beta disponible, les tests sont trop restrictifs")

        st.caption(
            "Remarque: Ainsi, on peut choisir un intervalle pour le coefficient. On estimera plus tard les modèles pour prédire le taux de défaut, et ceux-ci devront vérifier la condition énoncé par l'intervalle ici déclaré."
        )

with tab3:
    st.markdown("#### Corrélation des variables")
    macro_var_chosen = st.session_state.macro_kept_statio

    st.markdown("##### Calcul des variables explicatives retardées")
    st.caption(
        "On a maintenant des variables stationnaires. L'utilisateur peut choisir le nombre de retards pour les variables à variation trimestrielle et annuelle, mais il faut faire attention à ne pas créer trop de valeurs manquantes en début de période, surtout pour les variables annuelles."
    )

    col1, col2 = st.columns(2)
    default_value_ann = 2
    default_value_tri = 5

    with col1:
        number_lag_ann = st.number_input(
            "Choisir un nombre de retards pour les variables annuelles",
            min_value=0,
            max_value=8,
            value=default_value_ann,
        )
        st.write("Nombre de retards maximum :", number_lag_ann)

    with col2:
        number_lag_tri = st.number_input(
            "Choisir un nombre de retards pour les variables trimestrielles",
            min_value=0,
            max_value=8,
            value=default_value_tri,
        )
        st.write("Nombre de retards maximum :", number_lag_tri)

    number_lag_td = 1
    macro_lagged = compute_var_lagged(
        st.session_state.macro_kept_statio,
        number_lag_ann,
        number_lag_tri,
        number_lag_td,
    )
    macro_lagged_for_model = macro_lagged.copy()

    st.markdown(
        "##### Corrélation des variables explicatives retardées avec le taux de défaut"
    )

    corr_df = commpute_corr_matrix(macro_lagged)
    # On se focalise sur les correlations concernant le taux de défaut
    df_var_corr_to_DR = (
        pd.DataFrame(corr_df["td"])
        .rename(columns={"td": "Kendall Correlations"})
        .sort_values(by=["Kendall Correlations"])
        .drop("td", axis=0)
    )
    # peu importe le sens des signes, on cherche surtout à avoir les variables qui sont corrélées avec le taux de défaut : donc on prend la valeur absolue
    df_var_corr_to_DR["Kendall Correlations"] = df_var_corr_to_DR[
        "Kendall Correlations"
    ].abs()
    df_var_corr_to_DR = df_var_corr_to_DR.sort_values(
        by=["Kendall Correlations"], ascending=False
    )
    col1, col2 = st.columns(2)

    with col1:
        default_value_threshold = 0.15
        number_threshold = st.number_input(
            "Choisir un seuil minimal de corrélation pour utilser la variable",
            min_value=0.00,
            max_value=1.00,
            value=default_value_threshold,
        )
        st.write("Seuil =", number_threshold)

        list_var_in_model = choosing_seuil_corr_to_DR(
            df_var_corr_to_DR, number_threshold
        )
    with col2:
        st.dataframe(df_var_corr_to_DR.loc[list_var_in_model])
    ######################################################
    st.markdown("##### Corrélation des variables explicatives")
    st.write(
        "L'étude des corrélations permet d'appréhender la présomption de multicolinéarité dans le modèle. Il faut veiller à limiter celle-ci en supprimant des variables trop corrélées."
    )
    st.caption(
        "Ici on affiche les corrélations de toutes les variables, puis on permet à l'utilisateur de retirer des variables pour respecter un seuil de corrélation."
    )

    macro_lagged_seuil = macro_lagged.copy()
    macro_lagged_for_corr = macro_lagged.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Matrice initiale**")
        st.write(" ")
        corr_df_macro = commpute_corr_matrix_var_macro(
            macro_lagged, sorted(list_var_in_model)
        )

        fig, ax = plt.subplots()
        sns.heatmap(corr_df_macro.astype(float), cmap="coolwarm", center=0, annot=False)
        st.pyplot(fig)

    with col2:
        st.markdown("**Matrice post-sélection**")
        st.markdown("Choix du seuil:")
        # on définit le seuil de corrélation.
        default_seuil = 0.6
        number_seuil = st.number_input(
            "", min_value=0.00, max_value=1.00, value=default_seuil
        )

        first_elements_variables = [
            t[0] for t in find_high_corr_pairs(corr_df_macro, number_seuil)
        ]
        all_elements_variables = list(set(first_elements_variables))

        macro_lagged_for_corr = macro_lagged_for_corr[list_var_in_model].drop(
            all_elements_variables, axis=1
        )

        corr_df_for_corr = commpute_corr_matrix_var_macro(
            macro_lagged_for_corr, sorted(macro_lagged_for_corr.columns)
        )

        fig, ax = plt.subplots()
        sns.heatmap(
            corr_df_for_corr.astype(float), cmap="coolwarm", center=0, annot=False
        )
        st.pyplot(fig)

    with st.expander("Aperçu du Dataframe utilisé pour la modélisation"):
        st.dataframe(macro_lagged[corr_df_for_corr.columns])

    if "macro_model" not in st.session_state:
        st.session_state["macro_model"] = macro_lagged_for_model[
            list(corr_df_for_corr.columns) + ["td"]
        ]

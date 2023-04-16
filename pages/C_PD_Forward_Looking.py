################################   Packages    ###################################
import streamlit as st

from PIL import Image
import pandas as pd
import numpy as np
from streamlit import session_state
import sklearn.linear_model as lm




from scipy.stats import norm
from scipy import stats


import plotly.graph_objs as go



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
from functions_module import (
    model_OLS,
    split_train_test,
    interpretation_tests_st,
    graphique_periode_entrainement,
    tests_homoscedasticide_st,
    apply_same_preprocess_as_OLS,
    predict_test_OOS,
    predict_BMA,
    predict_test,
    train_and_evaluate_rf,
)
from functions_module import (
    preprocessing_global_for_prediction,
    preprocess_OLS_sort_exclude,
    preprocessing_additionnal_for_OLS_for_prediction,
    predict_test_BMA_OOS,
    rescale_process_l1l2,
    prepare_df_for_prediction,
    add_df_previsions_to_dict,
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
st.markdown("### PD Forward Looking")


tab1, tab2, tab3 = st.tabs(["Analyse descriptive", "Modèle de Merton-Vasicek", "PDFL"])


td = pd.read_excel("data/default_rate_quarterly.xlsx")
td_pred_bma = pd.read_csv("data/df_previsions_bma.csv")
td_pred_elastic = pd.read_csv("data/df_previsions_l1l2.csv")
matrice_ttc = pd.read_excel("data/matrice_moyenne.xlsx")
MMP = pd.read_excel("data/MMP.xlsx")
count = pd.read_excel("data/volumétrie.xlsx")
matrice_2010 = pd.read_excel("data/matrice2010.xlsx")
matrice_2018 = pd.read_excel("data/matrice2018.xlsx")
matrice_ttc = matrice_ttc.drop(columns="classe_prec")
MMP = MMP.drop(columns="classe_prec")
count = {
    2010: 2119555,
    2011: 2329088,
    2012: 2387998,
    2013: 2292403,
    2014: 2479271,
    2015: 2847368,
    2016: 3483698,
    2017: 4197345,
    2018: 4893004,
    2019: 5428672,
    2020: 5546968,
}
count = pd.Series(count)

# Initialisation des prédition de TD pour les 3 scénarios
td_pred_bma["Date"] = pd.to_datetime(td_pred_bma["Date"])
td_pred_bma["Année"] = td_pred_bma["Date"].dt.year
td_pred_bma = td_pred_bma.groupby("Année").mean()
td_pred_elastic["Date"] = pd.to_datetime(td_pred_elastic["Date"])
td_pred_elastic["Année"] = td_pred_elastic["Date"].dt.year
td_pred_elastic = td_pred_elastic.groupby("Année").mean()
td_adverse_bma = list(td_pred_bma["adverse"])
td_central_bma = list(td_pred_bma["central"])
td_favorable_bma = list(td_pred_bma["favorable"])
td_adverse_elastic = list(td_pred_elastic["adverse"])
td_central_elastic = list(td_pred_elastic["central"])
td_favorable_elastic = list(td_pred_elastic["favorable"])

## analyse descriptive

with tab1:

    def plot_TD_projections(td_pred_bma, nom_model):
        # Création des traces pour chaque colonne
        trace_central = go.Scatter(
            x=td_pred_bma.index, y=td_pred_bma["central"], name="Central"
        )
        trace_adverse = go.Scatter(
            x=td_pred_bma.index, y=td_pred_bma["adverse"], name="Adverse"
        )
        trace_favorable = go.Scatter(
            x=td_pred_bma.index, y=td_pred_bma["favorable"], name="Favorable"
        )

        # Mise en forme du layout du graphique
        layout = go.Layout(
            title="Evolution des taux de défauts projetés avec  {}".format(nom_model),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Valeurs"),
        )

        # Création de la figure et affichage du graphique
        fig = go.Figure(
            data=[trace_central, trace_adverse, trace_favorable], layout=layout
        )
        return fig

    st.markdown("#### Analyse descriptive")
    st.markdown("#")

    st.markdown("""##### Projections des défauts selon les modèles sélectionnés""")
    model_project_default_choice = st.radio(
        "Sélectionnez le modèle à projeter:",
        ["Régression linéaire pénalisé (l1,l2)", "BMA"],
    )
    if model_project_default_choice == "Régression linéaire pénalisé (l1,l2)":
        st.plotly_chart(
            plot_TD_projections(td_pred_elastic, "Régression linéaire pénalisé (l1,l2)")
        )
    else:
        st.plotly_chart(plot_TD_projections(td_pred_bma, "BMA"))

    st.markdown("""##### Evolution du nombre en contrepartie""")

    trace = go.Scatter(
        x=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        y=count,
        name="count",
    )

    layout = go.Layout(
        title="Evolution de la volumétrie",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Count"),
    )

    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)
    with st.expander("Matrice de migration moyenne"):
        st.dataframe(matrice_ttc)
    st.markdown("##### Résumé des matrices de migration")
    with st.expander("2010"):
        st.dataframe(matrice_2010)
    with st.expander("2019"):
        st.dataframe(matrice_2018)
    with st.expander("Matrice de migration: moyenne mobile pondérée"):
        st.dataframe(MMP)

    

# Merton
with tab2:
    st.markdown("#### Modèle de Merton")
    st.markdown("#")

    st.markdown("""##### Calibration des paramètres""")
    with st.expander("Rho", expanded=True):
        st.markdown("""**Formule:**""")
        st.markdown("""# """)
        st.latex(
            r"""
            \rho = \frac{V[\Phi^{-1}(DR^{historical})]}{1 + V[\Phi^{-1}(DR^{historical})]}"""
        )

        # Rho --> Mettre la formule 𝜌 = (𝕍[𝛷^(−1) (〖𝐷𝑅〗^ℎ𝑖𝑠𝑡𝑜𝑟𝑖𝑐𝑎𝑙)]) / (1 + 𝕍[𝛷^(−1) (〖𝐷𝑅〗^ℎ𝑖𝑠𝑡𝑜𝑟𝑖𝑐𝑎𝑙)])
        td_array = np.array(td["DR"])  # Vecteur des taux de défaut
        inv_norm_td = stats.norm.ppf(
            td_array
        )  # Fonction de répartition inverse de la distribution normale standard pour le vecteur td
        rho = np.var(inv_norm_td) / (1 + np.var(inv_norm_td))
        st.markdown("#### ")
        st.metric("Valeur estimée de rho", round(rho,3))

    with st.expander("DR", expanded=True):
        st.markdown("""**Taux de défaut TTC**""")
        DR_ttc = td["DR"].mean()
        st.markdown("#### ")
        st.metric("Valeur estimée de DR", str(round(DR_ttc,3)*100)+"%")

    with st.expander("Zt", expanded=True):
        st.markdown("**Zt**")
        # st.markdown("blabla")
        st.latex(
            r"""
z_t = \frac{\Phi^{-1} (\bar{DR}^{historical}) - \sqrt{(1 - \rho)} \times \Phi^{-1} (DR_t)}{\sqrt{\rho}}
"""
        )

        def cycle_vie_defaut(DR_TTC, td_array):
            # Calcul de la série Zt
            zt = (norm.ppf(DR_TTC) - np.sqrt(1 - rho) * norm.ppf(td_array)) / np.sqrt(
                rho
            )

            # Création de la trace pour la série Zt
            zt_trace = go.Scatter(
                x=np.array(td["Date"]), y=zt, name="Série Zt", mode="lines"
            )

            # Création de la figure
            fig = go.Figure(zt_trace)

            # Configuration de la mise en page
            fig.update_layout(
                title="Série du cycle de vie du défaut",
                xaxis_title="Année",
                yaxis_title="Série Zt",
            )

            # Affichage du graphique
            return fig

        st.plotly_chart(cycle_vie_defaut(DR_ttc, td_array))

    # Zt projétés

    def calculate_zt_proj(DR_pred1, DR_pred2, DR_pred3, rho, DR_ttc):
        zt_proj_1 = (
            norm.ppf(DR_ttc) - np.sqrt(1 - rho) * norm.ppf(DR_pred1)
        ) / np.sqrt(rho)
        zt_proj_2 = (
            norm.ppf(DR_ttc) - np.sqrt(1 - rho) * norm.ppf(DR_pred2)
        ) / np.sqrt(rho)
        zt_proj_3 = (
            norm.ppf(DR_ttc) - np.sqrt(1 - rho) * norm.ppf(DR_pred3)
        ) / np.sqrt(rho)
        return zt_proj_1, zt_proj_2, zt_proj_3

    z_proj_adverse_bma = calculate_zt_proj(
        td_adverse_bma[0], td_adverse_bma[1], td_adverse_bma[2], rho, DR_ttc
    )
    z_proj_central_bma = calculate_zt_proj(
        td_central_bma[0], td_central_bma[1], td_central_bma[2], rho, DR_ttc
    )
    z_proj_favorable_bma = calculate_zt_proj(
        td_favorable_bma[0], td_favorable_bma[1], td_favorable_bma[2], rho, DR_ttc
    )

    z_proj_adverse_elastic = calculate_zt_proj(
        td_adverse_elastic[0], td_adverse_elastic[1], td_adverse_elastic[2], rho, DR_ttc
    )
    z_proj_central_elastic = calculate_zt_proj(
        td_central_elastic[0], td_central_elastic[1], td_central_elastic[2], rho, DR_ttc
    )
    z_proj_favorable_elastic = calculate_zt_proj(
        td_favorable_elastic[0],
        td_favorable_elastic[1],
        td_favorable_elastic[2],
        rho,
        DR_ttc,
    )
    # Les années à afficher sur l'axe des x
    année_proj = ["2020", "2021", "2022"]

    # Création de la figure
    fig_bma_zt = go.Figure()

    # Ajout des trois séries de données
    fig_bma_zt.add_trace(
        go.Scatter(x=année_proj, y=z_proj_adverse_bma, name="Z adverse")
    )
    fig_bma_zt.add_trace(
        go.Scatter(x=année_proj, y=z_proj_central_bma, name="Z central")
    )
    fig_bma_zt.add_trace(
        go.Scatter(x=année_proj, y=z_proj_favorable_bma, name="Z favorable")
    )

    fig_bma_zt.update_layout(
        title="Z projetés avec le BMA", xaxis_title="années", yaxis_title="Série Zt"
    )

    # Affichage de la figure

    # Les années à afficher sur l'axe des x
    année_proj = ["2020", "2021", "2022"]

    # Création de la figure
    fig_l1_zt = go.Figure()

    # Ajout des trois séries de données
    fig_l1_zt.add_trace(
        go.Scatter(x=année_proj, y=z_proj_adverse_elastic, name="Z adverse")
    )
    fig_l1_zt.add_trace(
        go.Scatter(x=année_proj, y=z_proj_central_elastic, name="Z central")
    )
    fig_l1_zt.add_trace(
        go.Scatter(x=année_proj, y=z_proj_favorable_elastic, name="Z favorable")
    )

    fig_l1_zt.update_layout(
        title="Z projetés avec la régression linéaire pénalisé (L1,L2)",
        xaxis_title="années",
        yaxis_title="Série Zt",
    )

    # Affichage de la figure

    model_project_default_choice = st.radio(
        "Sélectionnez le modèle à projeter:",
        ["Régression linéaire pénalisé (l1,l2)", "BMA"],
        key=2,
    )
    if model_project_default_choice == "Régression linéaire pénalisé (l1,l2)":
        st.plotly_chart(fig_l1_zt)

    else:
        st.plotly_chart(fig_bma_zt)

    ## Fin analyse descriptive
    def forward_looking_cumulative_matrix(moyenne_migration, zt_proj, rho):
        n, m = moyenne_migration.shape
        if n != m:
            raise ValueError("La matrice de migration moyenne doit être carrée")

        # Initialisation de la matrice de migration forward-looking cumulée
        forward_looking = np.zeros((n, n))

        # Calcul de la probabilité de migration
        for i in range(n):
            for j in range(n):  # (i,n)
                forward_looking[i, j] = norm.cdf(
                    (1 / np.sqrt(1 - rho))
                    * (norm.ppf(moyenne_migration.iloc[i, j]) - np.sqrt(rho) * zt_proj)
                )

        # Normalisation des probabilités de migration par ligne

        forward_looking_df = pd.DataFrame(
            forward_looking,
            index=moyenne_migration.index,
            columns=moyenne_migration.columns,
        )

        return pd.DataFrame(forward_looking_df)

    matrice_adverse_1_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_adverse_bma[0], rho
    )
    matrice_adverse_2_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_adverse_bma[1], rho
    )
    matrice_adverse_3_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_adverse_bma[2], rho
    )

    matrice_central_1_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_central_bma[0], rho
    )
    matrice_central_2_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_central_bma[1], rho
    )
    matrice_central_3_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_central_bma[2], rho
    )

    matrice_favorable_1_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_favorable_bma[0], rho
    )
    matrice_favorable_2_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_favorable_bma[1], rho
    )
    matrice_favorable_3_bma_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_favorable_bma[2], rho
    )

    matrice_adverse_1_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_adverse_elastic[0], rho
    )
    matrice_adverse_2_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_adverse_elastic[1], rho
    )
    matrice_adverse_3_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_adverse_elastic[2], rho
    )

    matrice_central_1_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_central_elastic[0], rho
    )
    matrice_central_2_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_central_elastic[1], rho
    )
    matrice_central_3_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_central_elastic[2], rho
    )

    matrice_favorable_1_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_favorable_elastic[0], rho
    )
    matrice_favorable_2_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_favorable_elastic[1], rho
    )
    matrice_favorable_3_elastic_TTC = forward_looking_cumulative_matrix(
        matrice_ttc, z_proj_favorable_elastic[2], rho
    )

    matrice_adverse_1_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_adverse_bma[0], rho
    )
    matrice_adverse_2_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_adverse_bma[1], rho
    )
    matrice_adverse_3_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_adverse_bma[2], rho
    )

    matrice_central_1_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_central_bma[0], rho
    )
    matrice_central_2_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_central_bma[1], rho
    )
    matrice_central_3_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_central_bma[2], rho
    )

    matrice_favorable_1_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_favorable_bma[0], rho
    )
    matrice_favorable_2_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_favorable_bma[1], rho
    )
    matrice_favorable_3_bma_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_favorable_bma[2], rho
    )

    matrice_adverse_1_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_adverse_elastic[0], rho
    )
    matrice_adverse_2_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_adverse_elastic[1], rho
    )
    matrice_adverse_3_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_adverse_elastic[2], rho
    )

    matrice_central_1_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_central_elastic[0], rho
    )
    matrice_central_2_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_central_elastic[1], rho
    )
    matrice_central_3_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_central_elastic[2], rho
    )

    matrice_favorable_1_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_favorable_elastic[0], rho
    )
    matrice_favorable_2_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_favorable_elastic[1], rho
    )
    matrice_favorable_3_elastic_MMP = forward_looking_cumulative_matrix(
        MMP, z_proj_favorable_elastic[2], rho
    )

    def cumulative_matrix_3_years(m1, m2, m3, avg_matrix):
        # Initialisation de la matrice de migration forward-looking cumulée pour la première année
        cum_matrix_1 = m1.values

        # Calcul de la matrice de migration forward-looking cumulée pour la deuxième année
        cum_matrix_2 = np.dot(cum_matrix_1, m2)  # + cum_matrix_1

        # Calcul de la matrice de migration forward-looking cumulée pour la troisième année
        cum_matrix_3 = np.dot(cum_matrix_2, m3)  # + cum_matrix_2
        # Calcul du CEDF pour chaque année
        cedf_1 = pd.DataFrame(cum_matrix_1[:, -1], columns=["CEDF"])
        cedf_2 = pd.DataFrame(cum_matrix_2[:, -1], columns=["CEDF"])
        cedf_3 = pd.DataFrame(cum_matrix_3[:, -1], columns=["CEDF"])

        # Stockage de la dernière colonne de chaque matrice de migration cumulée dans une liste
        last_col_list = [cum_matrix_1[:, -1], cum_matrix_2[:, -1], cum_matrix_3[:, -1]]
        cum_matrices = [cum_matrix_1, cum_matrix_2, cum_matrix_3]

        for i in range(4, 10):
            # Calcul de la matrice de migration forward-looking cumulée pour la ième année
            cum_matrix_i = np.dot(
                cum_matrices[i - 2], avg_matrix
            )  # + cum_matrices[i-2]*0.1**(1.5)
            cum_matrices.append(cum_matrix_i)

            # Stockage de la dernière colonne de chaque matrice de migration cumulée dans la liste
            last_col_list.append(cum_matrix_i[:, -1])
        last_col_list = np.delete(last_col_list, -1, axis=1)
        return last_col_list  # cum_matrices, cedf_1, cedf_2, cedf_3

    PD_FL_adverse_bma_TTC = cumulative_matrix_3_years(
        matrice_adverse_1_bma_TTC,
        matrice_adverse_2_bma_TTC,
        matrice_adverse_3_bma_TTC,
        matrice_ttc,
    )
    PD_FL_central_bma_TTC = cumulative_matrix_3_years(
        matrice_central_1_bma_TTC,
        matrice_central_2_bma_TTC,
        matrice_central_3_bma_TTC,
        matrice_ttc,
    )
    PD_FL_favorable_bma_TTC = cumulative_matrix_3_years(
        matrice_favorable_1_bma_TTC,
        matrice_favorable_2_bma_TTC,
        matrice_favorable_3_bma_TTC,
        matrice_ttc,
    )

    PD_FL_adverse_bma_MMP = cumulative_matrix_3_years(
        matrice_adverse_1_bma_MMP,
        matrice_adverse_2_bma_MMP,
        matrice_adverse_3_bma_MMP,
        matrice_ttc,
    )
    PD_FL_central_bma_MMP = cumulative_matrix_3_years(
        matrice_central_1_bma_MMP,
        matrice_central_2_bma_MMP,
        matrice_central_3_bma_MMP,
        matrice_ttc,
    )
    PD_FL_favorable_bma_MMP = cumulative_matrix_3_years(
        matrice_favorable_1_bma_MMP,
        matrice_favorable_2_bma_MMP,
        matrice_favorable_3_bma_MMP,
        matrice_ttc,
    )

    PD_FL_adverse_elastic_TTC = cumulative_matrix_3_years(
        matrice_adverse_1_elastic_TTC,
        matrice_adverse_2_elastic_TTC,
        matrice_adverse_3_elastic_TTC,
        matrice_ttc,
    )
    PD_FL_central_elastic_TTC = cumulative_matrix_3_years(
        matrice_central_1_elastic_TTC,
        matrice_central_2_elastic_TTC,
        matrice_central_3_elastic_TTC,
        matrice_ttc,
    )
    PD_FL_favorable_elastic_TTC = cumulative_matrix_3_years(
        matrice_favorable_1_elastic_TTC,
        matrice_favorable_2_elastic_TTC,
        matrice_favorable_3_elastic_TTC,
        matrice_ttc,
    )

    PD_FL_adverse_elastic_MMP = cumulative_matrix_3_years(
        matrice_adverse_1_elastic_MMP,
        matrice_adverse_2_elastic_MMP,
        matrice_adverse_3_elastic_MMP,
        matrice_ttc,
    )
    PD_FL_central_elastic_MMP = cumulative_matrix_3_years(
        matrice_central_1_elastic_MMP,
        matrice_central_2_elastic_MMP,
        matrice_central_3_elastic_MMP,
        matrice_ttc,
    )
    PD_FL_favorable_elastic_MMP = cumulative_matrix_3_years(
        matrice_favorable_1_elastic_MMP,
        matrice_favorable_2_elastic_MMP,
        matrice_favorable_3_elastic_MMP,
        matrice_ttc,
    )

    def plot_PD_curves(PD_list, PD_MMP_list, label_solid, label_dashed, titre):
        PD_list = np.array(PD_list)
        PD_MMP_list = np.array(PD_MMP_list)

        # Création des couleurs en dégradé de rouge à vert
        colors = [
            f"rgb({int(max(0, 255-25*(i-1)))}, {int(min(255, 25*(i-1)))}, 0)"
            for i in range(11, 0, -1)
        ]

        # Création des traces pour PD_list
        PD_list_traces = []
        for i in range(1, 11):
            PD_list_traces.append(
                go.Scatter(
                    x=list(range(1, len(PD_list) + 1)),
                    y=PD_list[:, i - 1],
                    name=f"Rating {i} {label_solid} ",
                    mode="lines",
                    line=dict(color=colors[i - 1], width=2),
                )
            )

        # Création des traces pour PD_MMP_list
        PD_MMP_list_traces = []
        for i in range(1, 11):
            PD_MMP_list_traces.append(
                go.Scatter(
                    x=list(range(1, len(PD_MMP_list) + 1)),
                    y=PD_MMP_list[:, i - 1],
                    name=f"Rating {i} {label_dashed}",
                    mode="lines",
                    line=dict(color=colors[i - 1], dash="dash", width=2),
                )
            )

        # Création de la figure
        fig = go.Figure(PD_list_traces + PD_MMP_list_traces)

        # Configuration de la mise en page
        fig.update_layout(
            title=titre,  #'Evolution des Probabilité de défaut Forward Looking et FL MMP',
            xaxis_title="Années",
            yaxis_title="PD",
        )

        # Affichage du graphique
        return fig


with tab3:
    st.markdown("#### PDFL")
    st.markdown("Stress des matrices de migrations par rapport à la PDFL:")
    st.latex(
        r"p_{i,j}^{\mathrm{cum}}(z_{proj}) = \Phi\Big(\frac{1}{\sqrt{1-\rho}} \Big[ \Phi^{-1}(P_{i,j}) - \sqrt{\rho} z_{proj} \Big] \Big)"
    )
    tab1_,tab2_,tab3_=st.tabs(["PDFL comparaison scénarios","PDFL comparaison entre modèle","PDFL comparaison entre les méthodologies d'aggrégation de matrice sur le BMA"])
    

    with tab1_:
        st.markdown("##### PDFL comparaison scénarios")
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_adverse_bma_TTC,
                PD_FL_favorable_bma_TTC,
                "adverse",
                "favorable",
                "Evolution des Probabilité de défaut Forward Looking : adverse vs favorable",
            )
        )
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_adverse_bma_TTC,
                PD_FL_central_bma_TTC,
                "adverse",
                "central",
                "Evolution des Probabilité de défaut Forward Looking : adverse vs central",
            )
        )
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_central_bma_TTC,
                PD_FL_favorable_bma_TTC,
                "central",
                "favorable",
                "Evolution des Probabilité de défaut Forward Looking : central vs favorable",
            )
        )
    with tab2_:
        st.markdown("##### PDFL comparaison entre modèles")
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_adverse_bma_TTC,
                PD_FL_adverse_elastic_TTC,
                "BMA : adverse",
                "Elastic-net : adverse ",
                "Evolution des PD Forward Looking : BMA vs Régression linéaire (Elastic-net) : adverse",
            )
        )
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_central_bma_TTC,
                PD_FL_central_elastic_TTC,
                "BMA : adverse ",
                " Elastic-net : adverse ",
                "Evolution des PD Forward Looking : BMA vs Régression linéaire (Elastic-net) : central",
            )
        )
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_favorable_bma_TTC,
                PD_FL_favorable_elastic_TTC,
                "BMA : adverse",
                "Elastic-net : adverse",
                "Evolution des PD Forward Looking : BMA vs Régression linéaire (Elastic-net) : favorable",
            )
        )
    with tab3_:
        st.markdown("##### PDFL comparaison entre les méthodologies d'aggrégation de matrice sur le BMA")
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_adverse_bma_TTC,
                PD_FL_adverse_bma_MMP,
                "adverse : TTC",
                "adverse : MMP ",
                "Evolution des PD Forward Looking Méthdologie TTC vs MMP : adverse",
            )
        )
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_central_bma_TTC,
                PD_FL_central_bma_MMP,
                "central : TTC",
                "central : MMP ",
                "Evolution des PD Forward Looking Méthdologie TTC vs MMP : central",
            )
        )
        st.plotly_chart(
            plot_PD_curves(
                PD_FL_favorable_bma_TTC,
                PD_FL_favorable_bma_MMP,
                "favorable : TTC",
                "favorable : MMP ",
                "Evolution des PD Forward Looking Méthdologie TTC vs MMP : favorable",
            )
        )

    # PDFL comparaison entre les méthodo de matrice

################################   Packages    ###################################
import streamlit as st

from PIL import Image
import pandas as pd
import numpy as np
import sklearn.linear_model as lm

import statsmodels.api as sm

import plotly.express as px
import time

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import shap

import warnings

from pages.A_Traitement_et_Sélection_de_variables import macro_lagged_for_model

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


st.markdown("### Prévision du défaut")
macro_model = macro_lagged_for_model
df_bma = pd.read_csv("data/df_all_models_bma.csv", sep=",")
######## C'EST CE QU ON PREND VRAIMENT EN INPUT
df_lags = macro_model.copy()
df_lags = df_lags.dropna()

# df_lags = pd.read_csv('data/df_ready_for_regression.csv', sep=',')
# df_lags = df_lags.set_index("index")

df_train, df_test = split_train_test(
    df_lags
)  # on fait un split : 1 an pour le test, le reste sert à entraîner le modèle
date_fin_entrainement = "2017-10-31"

####################################   MODELE MCO   ######################################
###################################   entrainement   #####################################

with st.expander("Modèles", expanded=True):
    st.markdown("##### Modèles")
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Modèle MCO",
            "Modèle BMA (Bayesian model averaging)",
            "Modèle MCO avec pénalisation L1 L2",
            "Modèle Random Forest",
        ]
    )
    with tab1:
        st.markdown("###### Régression linéaire par MCO")
        variables_kept_signi = list(df_train.columns)
        variables_kept_in_model = list(df_train.columns)
        list_variable_a_retirer = st.multiselect(
            "On a retiré des variables par défaut pour le modèle MCO : Et pour cause elles n'étaient pas significatives. Pour pouvez ajouter/retirer des variables pour voir la différence avec le modèle MCO retenu:",
            variables_kept_signi,
            default=[
                "td",
                "UNR_tri_lag5",
                "RGDP_tri_lag5",
                "UNR_tri",
                "RREP_tri",
                "RREP_tri_lag4",
                "UNR_tri_lag1",
                "RGDP_tri_lag3",
                "RGDP_tri",
                "UNR_tri_lag3",
                "UNR_tri_lag4",
                "RREP_tri_lag1",
                "RREP_tri_lag3",
                "RREP_tri_lag2",
            ],
        )

        for variable in list_variable_a_retirer:
            if variable in variables_kept_in_model:
                variables_kept_in_model.remove(variable)

        row_signi, results_signi, dr_signi, var_signi = model_OLS(
            df_train, variables_kept_in_model
        )
        df_train_regression = df_train.copy()
        df_train_regression["td_predicted"] = results_signi.predict()

        ###################################   summary + hypotheses   #####################################
        # SUMMARY'S OLS ON DATAFRAME
        st.caption("Résultats des MCO")

        st.write(pd.DataFrame(results_signi.summary().tables[1]), width=800)
        show_datafr_corr_td_hypotheses_ols = st.checkbox(
            "Voir les tests des hypothèses de la régression:"
        )
        if show_datafr_corr_td_hypotheses_ols:
            # TESTS HOMOSCEDASTICITE
            tests_homoscedasticide_st(
                results_signi.resid,
                df_train_regression["td"],
                df_train_regression.drop("td", axis=1),
            )
            # HOMOSCEDASTICITE GRAPH
            fig = px.scatter(
                x=results_signi.predict(var_signi),
                y=results_signi.resid,
                labels={"x": "Valeurs prédites", "y": "Résidus"},
                title="Graphique de résidus pour montrer l homoscédasticité",
            )
            fig.add_hline(y=0, line_dash="dot", line_color="red")
            st.plotly_chart(fig)
            # TESTS GAUSSIEN
            interpretation_tests_st(row_signi)
            # CUSUM OLS RESIDUALS
            rls_reg = sm.RecursiveLS(pd.DataFrame(dr_signi), var_signi)
            res = rls_reg.fit()
            fig = res.plot_cusum(figsize=(10, 6))
            plt.xlabel("Observation")
            plt.ylabel("CUSUM")
            plt.title("CUSUM Chart")
            st.pyplot(fig)
            # QQ PLOT
            fig = sm.qqplot(results_signi.resid, line="r")
            plt.title("Q-Q plot for residus' OLS regression")
            plt.show()
            st.pyplot(fig)
            # Coefficient VIF
            st.write(
                "Le modèle admet {} variables avec un facteur VIF supérieur à 5".format(
                    row_signi["nb_var_vif_above_5"]
                )
            )

        ###########################   graphique sur train   ##############################
        # show graph of the estimation (train)
        graphique_periode_entrainement(
            df_train_regression,
            "le modèle sous-optimal",
            "td",
            "td_predicted",
            "streamlit",
            df_lags,
        )

        ###################################   test   #####################################

        predicted_y_ols = df_train.loc[
            df_train.index[-1], "td"
        ]  # initialisation de la prédiction : on prend la dernière observation de l'échantillon d'entraînement
        df_test_complete_ols = df_lags.drop("td", axis=1).copy()

        test_end_index_ols = df_test_complete_ols.index.get_loc(
            pd.to_datetime(date_fin_entrainement)
        )  # je vais prédire à partir de la fin de l'entrainement
        df_test_complete_ols = df_test_complete_ols.reset_index(drop=True)
        df_test_predict_ols = df_test_complete_ols.copy()

        df_test_complete_ols = apply_same_preprocess_as_OLS(
            df_test_complete_ols, variables_kept_in_model
        )
        df_test_predict_ols = apply_same_preprocess_as_OLS(
            df_test_predict_ols, variables_kept_in_model
        )

        df_test_predict_ols = predict_test_OOS(
            test_end_index_ols,
            df_test_complete_ols,
            df_test_predict_ols,
            results_signi,
            predicted_y_ols,
            df_lags,
        )

        graphique_periode_entrainement(
            df_test_predict_ols,
            "le modèle sous-optimal",
            "td_reel",
            "td_predict",
            "streamlit",
            df_lags,
        )
    ###################################   fin OLS   #####################################

    ####################################   MODELE BMA   ######################################
    #############   entrainement : on recupere les combinaisons deja obtenus   ###############

    with tab2:
        # Importer les modèles issus des combinaisons
        st.markdown("###### BMA")
        st.markdown("**Sélection de modèles**")
        st.metric("Nombre de modèles initial", df_bma.shape[0])
        ###################################   summary + hypotheses   #####################################
        show_datafr_corr_td_hypotheses_bma_1 = st.checkbox(
            "Voir les tests des hypothèses: "
        )
        if show_datafr_corr_td_hypotheses_bma_1:
            # HISTOGRAMME COEFFICIENTS LAG 1 DU TD
            df_filtered = df_bma[df_bma["td_lag1_param"].notna()]
            fig = px.histogram(
                df_filtered, x="td_lag1_param", color_discrete_sequence=["plum"]
            )
            fig.update_layout(
                xaxis_title="td_lag1_param",
                yaxis_title="Frequency",
                title="Histogram of td_lag1_param",
            )
            st.plotly_chart(fig)

            # TESTS RESIDUS GAUSSIENS
            counts = (
                df_bma[["boolean_shapiro", "boolean_kolmogorov", "boolean_anderson"]]
                .apply(lambda row: row.value_counts()[True], axis=1)
                .value_counts()
                .sort_index()
            )
            colors = [
                "rgba(166, 206, 227, 0.6)",
                "rgba(31, 120, 180, 0.6)",
                "rgba(52, 40, 145, 0.8)",
                "rgba(51, 160, 44, 0.6)",
            ]
            fig = go.Figure(
                go.Bar(x=counts.index, y=counts.values, marker_color=colors)
            )
            fig.update_layout(
                title="Nombre de modèles qui passent X tests",
                xaxis_title="Nombre de tests qui rejettent l'hypothèses de normalité",
                yaxis_title="Nombre de modèles MCO",
            )
            st.plotly_chart(fig)

            # TESTS HOMOSCEDASTICITE
            counts = (
                df_bma[
                    [
                        "boolean_breuschpagan",
                        "boolean_white",
                        "boolean_goldfeldquandt",
                    ]
                ]
                .apply(lambda row: sum(row), axis=1)
                .value_counts()
                .sort_index()
            )
            colors = [
                "rgba(187, 0, 116, 1)",
                "rgba(225, 0, 109, 1)",
                "rgba(255, 0, 109, 1)",
                "rgba(255, 0, 255, 1)",
            ]

            fig = go.Figure(
                go.Bar(x=counts.index, y=counts.values, marker_color=colors)
            )
            fig.update_layout(
                title="Visualisation des modèles qui respectent les hypothèses d'homoscédasticité des résidus.",
                xaxis_title="Nombre de tests qui rejettent l'hypothèse H0. (x=3 <=> Tous les modèles rejette l'hypothèse d'homoscédasticité)",
                yaxis_title="Nombre de modèles",
            )
            st.plotly_chart(fig)

            colors = [
                "rgba(166, 206, 227, 0.6)",
                "rgba(31, 120, 180, 0.6)",
                "rgba(52, 40, 145, 0.8)",
                "rgba(51, 160, 44, 0.6)",
            ]
            fig = go.Figure(
                go.Histogram(x=df_bma["nb_var_vif_above_5"], marker_color=colors)
            )
            fig.update_layout(
                title="Nombre de modèles qui contiennent X variables avec un VIF supérieur à 5",
                xaxis_title="Nombre de variables avec VIF > 5",
                yaxis_title="Count",
            )
            st.plotly_chart(fig)

        # RETIRER LES MODELES QUI NE RESPECTENT PAS LES HYPOTHESES EN MAJORITE
        percentages_heterosc = df_bma[
            ["boolean_breuschpagan", "boolean_white", "boolean_goldfeldquandt"]
        ].apply(lambda row: sum(row) / len(row), axis=1)
        df_bma["perc_of_True_heterosc"] = percentages_heterosc
        percentages_non_gauss = df_bma[
            ["boolean_shapiro", "boolean_kolmogorov", "boolean_anderson"]
        ].apply(lambda row: sum(row) / len(row), axis=1)
        df_bma["perc_of_True_non_gauss"] = percentages_non_gauss
        # df_bma[['perc_of_True_heterosc', 'boolean_breuschpagan', 'boolean_white', 'boolean_goldfeldquandt', 'perc_of_True_non_gauss', 'boolean_shapiro', 'boolean_kolmogorov', 'boolean_anderson']]
        df_bma_hypo_valid = df_bma[
            (df_bma["perc_of_True_heterosc"] < 0.5)
            & (df_bma["perc_of_True_non_gauss"] < 0.5)
            & (df_bma["nb_var_vif_above_5"] < 1)
        ]
        df_bma_hypo_valid = df_bma_hypo_valid.drop(
            [
                "perc_of_True_heterosc",
                "perc_of_True_non_gauss",
                "nb_var_vif_above_5",
                "boolean_breuschpagan",
                "boolean_white",
                "boolean_goldfeldquandt",
                "boolean_shapiro",
                "boolean_kolmogorov",
                "boolean_anderson",
            ],
            axis=1,
        )

        # legères stats descriptives avant de retirer les modèles avec des variables non-significatives
        st.metric("Nombre de modèles post-tests", df_bma_hypo_valid.shape[0])

        show_datafr_corr_td_hypotheses_bma_2 = st.checkbox(
            "Voir les tests des hypothèses (Significativité des variables):"
        )
        if show_datafr_corr_td_hypotheses_bma_2:
            st.markdown(
                "**Vérifier la significativité des variables dans tous les sous-modèles.**"
            )
            pval_cols = [
                col for col in df_bma_hypo_valid.columns if col.endswith("_param")
            ]
            for col in pval_cols:
                st.write(
                    col,
                    ": Nombre de modèles sans cette variables :",
                    df_bma_hypo_valid[col].isna().sum(),
                )

        # RETIRER LES MODELES AVEC UN MOINS UNE VARIABLE NON SIGNIFICATIVE
        pval_cols = [col for col in df_bma_hypo_valid.columns if col.endswith("_pval")]
        for col in pval_cols:
            df_bma_hypo_valid[col] = df_bma_hypo_valid[col].fillna(
                -100
            )  # ca facilite le mask juste après.
        mask = (df_bma_hypo_valid[pval_cols] < 0.05).all(axis=1)
        df_significant = df_bma_hypo_valid[mask]
        st.metric("Nombre de modèles à soumettre au BMA.", df_significant.shape[0])

        st.markdown("**Construction du BMA.**")
        st.markdown("- On garde les meilleurs modèles selon le U de Theil")
        df_significant = df_significant.sort_values(by=["U de Theil"])
        n = len(df_significant) // 2
        df_significant = df_significant.head(n)

        st.markdown(
            "- On calcule les probabilités a priori, a posteriori. Puis on agrège les modèles."
        )

        print(df_significant["bic"])

        def calculate_prob_post_num(x):
            return np.exp(-0.5 * x)

        df_significant["prob_post_num"] = df_significant[["bic"]].applymap(
            calculate_prob_post_num
        )
        df_significant["prob_posteriori"] = df_significant["prob_post_num"].apply(
            lambda x: x / df_significant["prob_post_num"].sum()
        )

        ### Création du dataframe avec les colonnes que l'on garde car au moins un modèle admet un coefficient pour la variable & aussi on applique la pondération des coefficients par la probabilité a posteriori du modele. On fillna par 0 les colonnes vides
        # filter out the columns that end with '_param'
        param_cols = [col for col in df_significant.columns if col.endswith("_param")]

        # create a new dataframe with the filtered columns multiplied by 'prob_posteriori'
        df_significant_param = df_significant[param_cols].multiply(
            df_significant["prob_posteriori"], axis=0
        )

        # print the new dataframe
        df_significant_param = df_significant_param.fillna(0)

        ### On drop les colonnes sans coefficients ; ce sont des variables non significatives ou qui n'expliquent pas la target.
        # create a list of columns to drop
        cols_to_drop = []

        # loop through each column in the DataFrame
        for col in df_significant_param.columns:
            # check if the column is filled with 0
            if df_significant_param[col].eq(0).all():
                cols_to_drop.append(col)

        # drop the columns that are filled with 0
        df_significant_param_no_0 = df_significant_param.drop(cols_to_drop, axis=1)

        ### Finalement on va sommer les colonnes, ca fait qu'on obtient dans chaque colonne le paramètre de la variable associée
        # sum all the columns of the DataFrame
        df_sum = df_significant_param_no_0.sum()

        # convert the resulting series to a DataFrame with one row
        df_row_parameters = pd.DataFrame(df_sum).transpose()

        # Create a dictionary of old and new column names
        rename_dict = {
            col: col.replace("_param", "") for col in df_row_parameters.columns
        }

        # Rename the columns using the dictionary
        df_row_parameters = df_row_parameters.rename(columns=rename_dict)
        st.markdown("Les coefficients issus du BMA sont les suivants:")
        st.dataframe(df_row_parameters)

        ###########################   graphique sur train   ##############################
        st.markdown("**Echantillon d'entraînement du BMA**")

        pred_values_BMA = predict_BMA(
            df_row_parameters, df_train.drop("td", axis=1), df_row_parameters
        )

        df_pred = pd.DataFrame(
            pred_values_BMA, index=df_train.index, columns=["pred_values_BMA"]
        )
        y_obs = pd.DataFrame(df_train["td"])
        df_actual_and_predict = pd.merge(
            df_pred, y_obs, how="inner", left_index=True, right_index=True
        )
        x = df_actual_and_predict.index
        trace_pred = go.Scatter(
            x=df_actual_and_predict.index,
            y=df_actual_and_predict["pred_values_BMA"],
            name='Predicted "DRt estimé"',
            line=dict(color="blue"),
        )
        trace_actual = go.Scatter(
            x=df_actual_and_predict.index,
            y=df_actual_and_predict["td"],
            name='Actual "DRt"',
            line=dict(color="red"),
        )
        layout = go.Layout(
            title="Predicted vs Actual DRt Values",
            xaxis=dict(title="Index"),
            yaxis=dict(title="Value"),
        )

        RMSE_model = round(
            np.sqrt(
                mean_squared_error(
                    df_actual_and_predict["pred_values_BMA"].to_numpy(),
                    df_actual_and_predict["td"].to_numpy(),
                )
            ),
            4,
        )
        fig = go.Figure(data=[trace_pred, trace_actual], layout=layout)
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text="La Root Mean Squared Error sur l'échantillon ci-dessous est du : "
            + str(RMSE_model),
            showarrow=False,
            font=dict(size=14),
        )
        st.plotly_chart(fig)

        ###################################   test   #####################################

        ### Initialisation de la boucle prédictive/itérative et création du dataframe de test qui acceuille les prédictions
        predicted_y_bma = df_train.loc[
            df_train.index[-1], "td"
        ]  # initialisation de la prédiction : on prend la dernière observation de l'échantillon d'entraînement
        df_test_complete_bma = df_lags.copy()
        test_end_index_bma = df_test_complete_bma.index.get_loc(
            pd.to_datetime(date_fin_entrainement)
        )  # je vais prédire à partir de la fin de l'entrainement
        df_test_complete_bma = df_test_complete_bma.reset_index(drop=True)
        df_test_predict_bma = df_test_complete_bma.copy()

        for i in range(
            test_end_index_bma, len(df_test_complete_bma.drop("td", axis=1))
        ):
            # Add predicted value for previous date to predictors
            try:
                df_test_complete_i = df_test_complete_bma.drop("td", axis=1).iloc[[i]]

            except:
                df_test_complete_bma.drop("td", axis=1).loc[
                    i, "td_lag1"
                ] = predicted_y_bma
                df_test_complete_i = df_test_complete_bma.drop("td", axis=1).iloc[[i]]
            ################################# PREDICTION ########################################
            # Make prediction for current date
            predicted_y_i = predict_BMA(
                df_row_parameters, df_test_complete_i, df_row_parameters
            )

            ################################# PREDICTION ########################################
            df_test_predict_bma.loc[
                df_test_complete_bma.drop("td", axis=1).index[i], "td_predict"
            ] = predicted_y_bma

            # Update predicted value for next date
            try:
                predicted_y_bma = predicted_y_i.values[0]
            except:
                predicted_y_bma = predicted_y_i[0]
        df_lags_td = df_lags["td"].reset_index(
            drop=True
        )  # utilisation de la colonne avec tous les DR connus...
        df_test_predict_bma[
            "td_reel"
        ] = df_lags_td  # pour ensuite les comparer dans le dataframe avec td-estimé.

        graphique_periode_entrainement(
            df_test_predict_bma,
            "le modèle BMA",
            "td_reel",
            "td_predict",
            "streamlit",
            df_lags,
        )

    ####################################   MODELE L1L2   ######################################
    ###############   entrainement : étape à savoir : normalisation, split   #################

    with tab3:
        # Normalisation et split du dataframe
        st.markdown("###### Régression linéaire avec pénalité de type L1/L2")
        scaler_variables = StandardScaler()
        df_variables_l1l2 = pd.DataFrame(
            scaler_variables.fit_transform(df_lags.drop("td", axis=1)),
            columns=df_lags.drop("td", axis=1).columns,
            index=df_lags.index,
        )
        scaler_td = StandardScaler()
        df_td_l1l2 = pd.DataFrame(
            scaler_td.fit_transform(df_lags[["td"]]), index=df_lags["td"].index
        )
        df_lags_l1l2 = df_variables_l1l2.copy()
        df_lags_l1l2[
            "td"
        ] = df_td_l1l2  # pour ensuite les comparer dans le dataframe avec td-estimé.
        df_train_l1l2, df_test_l1l2 = split_train_test(df_lags_l1l2)

        # entrainement du modèle
        l1l2 = lm.ElasticNet(alpha=0.025, l1_ratio=0.8).fit(
            df_train_l1l2.drop("td", axis=1), df_train_l1l2["td"]
        )

        # prediction sur l'entrainement et teste : AVEC la transformation scaler inverse
        train_cible = pd.DataFrame(scaler_td.inverse_transform(df_train_l1l2[["td"]]))
        train_prediction_unscaled = pd.DataFrame(
            scaler_td.inverse_transform(
                pd.DataFrame(l1l2.predict(df_train_l1l2.drop("td", axis=1)))
            )
        )
        test_cible = pd.DataFrame(scaler_td.inverse_transform(df_test_l1l2[["td"]]))
        test_prediction_unscaled = pd.DataFrame(
            scaler_td.inverse_transform(
                pd.DataFrame(l1l2.predict(df_test_l1l2.drop("td", axis=1)))
            )
        )

        # dataframe avec les coefficients
        data = {
            df_train.drop("td", axis=1).columns[i]: l1l2.coef_[i]
            for i in range(len(df_train.drop("td", axis=1).columns))
        }
        df_coef = pd.DataFrame(data, index=["Coefficients"])
        st.markdown("**Les coefficients de la régression OLS avec pénalités:**")
        st.dataframe(df_coef)

        ###################################   summary + hypotheses   #####################################
        show_datafr_corr_td_hypotheses_ols_l1l2 = st.checkbox(
            "Voir les tests des hypothèses du modèle pénalisé:"
        )
        if show_datafr_corr_td_hypotheses_ols_l1l2:
            l1l2_resid = train_prediction_unscaled - train_cible
            # TESTS HOMOSCEDASTICITE
            tests_homoscedasticide_st(
                l1l2_resid, df_train_l1l2[["td"]], df_train_l1l2.drop("td", axis=1)
            )
            # CUSUM OLS RESIDUALS
            rls_reg = sm.RecursiveLS(train_cible, train_prediction_unscaled)
            res = rls_reg.fit()
            fig = res.plot_cusum(figsize=(10, 6))
            plt.xlabel("Observation")
            plt.ylabel("CUSUM")
            plt.title("CUSUM Chart")
            st.pyplot(fig)
            # QQ PLOT
            fig = sm.qqplot(l1l2_resid.values, line="r")
            plt.title("Q-Q plot for residus' OLS regression")
            plt.show()
            st.pyplot(fig)

        ###########################   graphique sur train   ##############################
        ### Initialisation de la boucle prédictive/itérative et création du dataframe de test qui acceuille les prédictions
        df_train_predict_l1l2 = pd.DataFrame()
        df_train_predict_l1l2["td_predict"] = train_prediction_unscaled
        df_train_predict_l1l2["td_reel"] = scaler_td.inverse_transform(
            df_train_l1l2[["td"]]
        )

        graphique_periode_entrainement(
            df_train_predict_l1l2,
            "le modèle OLS avec pénalité L1L2",
            "td_reel",
            "td_predict",
            "streamlit",
            df_lags,
        )

        ###################################   test   #####################################

        ### Initialisation de la boucle prédictive/itérative et création du dataframe de test qui acceuille les prédictions
        predicted_y = df_train_l1l2.loc[
            df_train_l1l2.index[-1], "td"
        ]  # initialisation de la prédiction : on prend la dernière observation de l'échantillon d'entraînement
        df_test_complete = df_lags_l1l2.drop("td", axis=1).copy()
        test_end_index = df_test_complete.index.get_loc(
            pd.to_datetime(date_fin_entrainement)
        )  # je vais prédire à partir de la fin de l'entrainement
        df_test_complete = df_test_complete.reset_index(drop=True)
        df_test_predict = df_test_complete.copy()

        ### on prédit par itération
        elastic_net_predict = predict_test(
            test_end_index,
            df_test_complete,
            df_test_predict,
            l1l2,
            predicted_y,
            df_lags,
        )

        ### scaler inverse
        elastic_net_predict_variables = pd.DataFrame(
            scaler_variables.inverse_transform(
                elastic_net_predict.drop(["td_predict", "td_reel"], axis=1)
            ),
            columns=elastic_net_predict.drop(["td_predict", "td_reel"], axis=1).columns,
        )
        df_td_predict = pd.DataFrame(
            scaler_td.inverse_transform(elastic_net_predict[["td_predict"]])
        )

        elastic_net_results_on_cible = pd.DataFrame()
        elastic_net_results_on_cible["td_predict"] = df_td_predict
        elastic_net_results_on_cible["td_reel"] = elastic_net_predict[
            ["td_reel"]
        ].values
        elastic_net_results_on_cible["td_cible"] = df_lags[["td"]].values

        graphique_periode_entrainement(
            elastic_net_results_on_cible,
            "le modèle Elastic Net",
            "td_reel",
            "td_predict",
            "streamlit",
            df_lags,
        )

    ##################################   MODELE RANDOM FOREST   ###############################
    ###########################   entrainement from scratch du RF   ###########################

    with tab4:
        n_estimators = [10, 50, 100]
        max_depth = [2, 5, 10]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]

        results_df_rf = pd.DataFrame(
            columns=[
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "train_rmse",
                "test_rmse",
            ]
        )

        # Loop over all hyperparameter combinations and train/evaluate a random forest regressor for each one
        for n in n_estimators:
            for d in max_depth:
                for s in min_samples_split:
                    for l in min_samples_leaf:
                        train_rmse, test_rmse = train_and_evaluate_rf(
                            n,
                            d,
                            s,
                            l,
                            df_train.drop("td", axis=1),
                            df_train["td"],
                            df_test.drop("td", axis=1),
                            df_test["td"],
                        )
                        print(
                            f"n_estimators={n}, max_depth={d}, min_samples_split={s}, min_samples_leaf={l}, train_rmse={train_rmse}, test_rmse={test_rmse}"
                        )
                        results_df_rf = results_df_rf.append(
                            {
                                "n_estimators": n,
                                "max_depth": d,
                                "min_samples_split": s,
                                "min_samples_leaf": l,
                                "train_rmse": train_rmse,
                                "test_rmse": test_rmse,
                            },
                            ignore_index=True,
                        )
        results_df_rf["difference_train_test"] = (
            results_df_rf["train_rmse"] - results_df_rf["test_rmse"]
        )
        results_df_rf = results_df_rf.sort_values("difference_train_test")
        results_df_rf = results_df_rf.reset_index(drop=True)
        best_model_rf = (
            results_df_rf[
                (results_df_rf["difference_train_test"] < 0)
                & (results_df_rf["difference_train_test"] > -0.0018)
            ]
            .sort_values(by="train_rmse")
            .iloc[0]
        )

        # Train the best model on the full training set
        X_train_rf = df_train.drop("td", axis=1)
        y_train_rf = df_train["td"]
        rf = RandomForestRegressor(
            n_estimators=best_model_rf["n_estimators"].astype(int),
            max_depth=best_model_rf["max_depth"].astype(int),
            min_samples_split=best_model_rf["min_samples_split"].astype(int),
            min_samples_leaf=best_model_rf["min_samples_leaf"].astype(int),
        )
        rf.fit(X_train_rf, y_train_rf)

        # Make predictions on the test set using the best model
        X_test_rf = df_test.drop("td", axis=1)
        y_test_rf = df_test["td"]
        y_test_pred_rf = rf.predict(X_test_rf)

        ###################################   SHAP   #####################################

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_train_rf)
        fig, ax = plt.subplots(figsize=(4, 4))
        summary_plot = shap.summary_plot(
            shap_values, X_train_rf, show=False, max_display=10
        )
        ax.set_title("Graphique des valeurs de SHAP")
        plt.tight_layout()
        st.pyplot(fig)

        ###########################   graphique sur train   ##############################
        X_train_rf["td_predicted"] = rf.predict(X_train_rf)
        X_train_rf["td"] = y_train_rf.values
        graphique_periode_entrainement(
            X_train_rf, "le modèle RF", "td", "td_predicted", "streamlit", df_lags
        )

        ###################################   test   #####################################

        ### Initialisation de la boucle prédictive/itérative et création du dataframe de test qui acceuille les prédictions
        # Preparation du dataframe pour le test : initialisation pour la fonction predict_test
        predicted_y_rf = df_train.loc[
            df_train.index[-1], "td"
        ]  # initialisation de la prédiction : on prend la dernière observation de l'échantillon d'entraînement
        df_test_complete_rf = df_lags.copy()
        test_end_index_rf = df_test_complete_rf.index.get_loc(
            pd.to_datetime(date_fin_entrainement)
        )  # je vais prédire à partir de la fin de l'entrainement
        df_test_complete_rf = df_test_complete_rf.reset_index(drop=True)
        df_test_predict_rf = df_test_complete_rf.copy()
        # Prediction Out of sample
        df_test_predict_rf = predict_test(
            test_end_index,
            df_test_complete_rf.drop("td", axis=1),
            df_test_predict_rf,
            rf,
            predicted_y_rf,
            df_lags,
        )

        graphique_periode_entrainement(
            df_test_predict_rf,
            "le modèle Random Forest",
            "td_reel",
            "td_predict",
            "streamlit",
            df_lags,
        )

##################################   Graphique all models on test set   ###############################
with st.expander("Performance de l'ensemble des modèles", expanded=True):
    st.markdown("##### Performance de l'ensemble des modèles")
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_lags.index, y=df_train_regression['td_predicted'], mode='lines', name='MCO', line=dict(color='red')))
    fig.add_trace(
        go.Scatter(
            x=df_lags.index,
            y=df_lags["td"],
            mode="lines",
            name="DRt observé",
            line=dict(color="red"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_lags.index,
            y=df_test_predict_ols["td_predict"],
            mode="lines",
            name="MCO",
            line=dict(color="darkgrey"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_lags.index,
            y=df_test_predict_bma["td_predict"],
            mode="lines",
            name="BMA",
            line=dict(color="lime"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_lags.index,
            y=elastic_net_results_on_cible["td_predict"],
            mode="lines",
            name="L1L2",
            line=dict(color="darkorange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_lags.index,
            y=df_test_predict_rf["td_predict"],
            mode="lines",
            name="RF",
            line=dict(color="midnightblue"),
        )
    )
    # Add axis labels and a title
    fig.update_layout(
        xaxis_title="Dates",
        yaxis_title="Taux (%)",
        title="Graphique pour représenter tous les modèles sur la période de Test (2018-2019)",
    )
    st.plotly_chart(fig)

##############################   Prediction 2020 2021 2022   ##############################
###########################   importation et cleaning general   ###########################
with st.expander("Prévisions & Scénarios", expanded=True):
    st.markdown("##### Prévisions & Scénarios")

    df_previsions_dict = (
        {}
    )  # dictionnaire qui va réportorier toutes les prévisions des modèles

    # importation des sets complets (2010 - 2019)
    td = pd.read_excel("data/default_rate_quarterly.xlsx")
    td = td.set_index("Date")
    macro_train_test = pd.read_excel("data/variables_macroeconomiques.xlsx")
    macro_train_test = macro_train_test.set_index("Date")
    # importation des sets des scenarios (2020 - 2022)
    macro_adverse = pd.read_excel("data/scenario_adverse_bce_20_22.xlsx")
    macro_adverse = macro_adverse.set_index("Date")
    macro_baseline = pd.read_excel("data/scenario_baseline_bce_20_22.xlsx")
    macro_baseline = macro_baseline.set_index("Date")
    macro_favorable = pd.read_excel("data/scenario_favorable_bce_20_22.xlsx")
    macro_favorable = macro_favorable.set_index("Date")
    # Indexer partout par la date du taux de defaut
    macro_adverse["td"] = td
    macro_baseline["td"] = td
    macro_favorable["td"] = td
    macro_train_test["td"] = td
    # appliquer le preprocessing habituel : choix par defaut ici. /!\ On devrait reprendre les mêmes choix que sur la page I quand on choisit quelles variables sont transformées
    list_col_saison = ["HICP"]
    macro_train_test = remove_season(macro_train_test, list_col_saison)
    lissage_moyenne_mobile(macro_train_test, ["RGDP", "RREP"])
    # concaténation des dataframes pour avoir un dataframe par scenario (sachant que les périodes 2010-2019 sont équivalentes pour chaque scenario)
    macro_train_test_head = macro_train_test.head(44)
    macro_baseline = pd.concat([macro_train_test_head, macro_baseline])
    macro_baseline.loc[macro_train_test_head.index, :] = macro_train_test
    macro_favorable = pd.concat([macro_train_test_head, macro_favorable])
    macro_favorable.loc[macro_train_test_head.index, :] = macro_train_test
    macro_adverse = pd.concat([macro_train_test_head, macro_adverse])
    macro_adverse.loc[macro_train_test.index, :] = macro_train_test

    # Proprocessing pour toutes les séries, pour tous les modèles : retarder les varaibles, enlever les variables pas stationnaires...
    macro_lagged_for_model_adv = preprocessing_global_for_prediction(
        macro_adverse, df_lags
    )
    macro_lagged_for_model_baseline = preprocessing_global_for_prediction(
        macro_baseline, df_lags
    )
    macro_lagged_for_model_favorable = preprocessing_global_for_prediction(
        macro_favorable, df_lags
    )

    model_showed = st.radio(
        "Choix du modèle qu'on souhaite observer:", ("OLS", "BMA", "L1L2", "RF")
    )

    if model_showed == "OLS":
        model_chosen = results_signi
        need_it_ = "yes_ols"

    elif model_showed == "L1L2":
        model_chosen = l1l2
        need_it_ = "yes_l1l2"

    elif model_showed == "BMA":
        model_chosen = "BMA"
        need_it_ = "yes_BMA"
    else:
        need_it_ = "no"
        model_chosen = rf

    if need_it_ == "yes_ols":
        macro_lagged_for_model_adv = preprocess_OLS_sort_exclude(
            macro_lagged_for_model_adv, need_it_, list_variable_a_retirer
        )
        macro_lagged_for_model_baseline = preprocess_OLS_sort_exclude(
            macro_lagged_for_model_baseline, need_it_, list_variable_a_retirer
        )
        macro_lagged_for_model_favorable = preprocess_OLS_sort_exclude(
            macro_lagged_for_model_favorable, need_it_, list_variable_a_retirer
        )

    if need_it_ == "yes_l1l2":
        macro_td_l1l2 = pd.DataFrame(
            scaler_td.fit_transform(macro_lagged_for_model_adv[["td"]]),
            index=macro_lagged_for_model_adv["td"].index,
        )
        macro_lagged_for_model_adv = pd.DataFrame(
            scaler_variables.fit_transform(
                macro_lagged_for_model_adv.drop("td", axis=1)
            ),
            columns=macro_lagged_for_model_adv.drop("td", axis=1).columns,
            index=macro_lagged_for_model_adv.index,
        )
        macro_lagged_for_model_baseline = pd.DataFrame(
            scaler_variables.fit_transform(
                macro_lagged_for_model_baseline.drop("td", axis=1)
            ),
            columns=macro_lagged_for_model_baseline.drop("td", axis=1).columns,
            index=macro_lagged_for_model_baseline.index,
        )
        macro_lagged_for_model_favorable = pd.DataFrame(
            scaler_variables.fit_transform(
                macro_lagged_for_model_favorable.drop("td", axis=1)
            ),
            columns=macro_lagged_for_model_favorable.drop("td", axis=1).columns,
            index=macro_lagged_for_model_favorable.index,
        )

        macro_lagged_for_model_adv["td"] = macro_td_l1l2.values
        macro_lagged_for_model_baseline["td"] = macro_td_l1l2.values
        macro_lagged_for_model_favorable["td"] = macro_td_l1l2.values

    ### Initialisation de la boucle prédictive/itérative et création du dataframe de test qui acceuille les prédictions
    date_fin_out_of_sample = "2019-10-31"
    predicted_y = macro_lagged_for_model_adv.loc[
        "2019-10-31", "td"
    ]  # initialisation de la prédiction : on prend la dernière observation de l'échantillon d'entraînement

    # preparer les inputs pour la prediction, c'est pour initialiser les predictions
    (
        df_test_complete_adv,
        test_end_index_adv,
        df_test_predict_adv,
    ) = prepare_df_for_prediction(macro_lagged_for_model_adv, date_fin_out_of_sample)
    (
        df_test_complete_baseline,
        test_end_index_baseline,
        df_test_predict_baseline,
    ) = prepare_df_for_prediction(
        macro_lagged_for_model_baseline, date_fin_out_of_sample
    )
    (
        df_test_complete_favorable,
        test_end_index_favorable,
        df_test_predict_favorable,
    ) = prepare_df_for_prediction(
        macro_lagged_for_model_favorable, date_fin_out_of_sample
    )

    # prediction
    if need_it_ == "yes_BMA":
        df_test_predict_adv = predict_test_BMA_OOS(
            test_end_index_adv,
            df_test_complete_adv,
            df_test_predict_adv,
            df_row_parameters,
            predicted_y,
            macro_lagged_for_model_adv,
        )
        df_test_predict_baseline = predict_test_BMA_OOS(
            test_end_index_baseline,
            df_test_complete_baseline,
            df_test_predict_baseline,
            df_row_parameters,
            predicted_y,
            macro_lagged_for_model_baseline,
        )
        df_test_predict_favorable = predict_test_BMA_OOS(
            test_end_index_favorable,
            df_test_complete_favorable,
            df_test_predict_favorable,
            df_row_parameters,
            predicted_y,
            macro_lagged_for_model_favorable,
        )
    else:
        df_test_predict_adv = predict_test_OOS(
            test_end_index_adv,
            df_test_complete_adv.drop("td", axis=1),
            df_test_predict_adv,
            model_chosen,
            predicted_y,
            macro_lagged_for_model_adv,
        )
        df_test_predict_baseline = predict_test_OOS(
            test_end_index_baseline,
            df_test_complete_baseline.drop("td", axis=1),
            df_test_predict_baseline,
            model_chosen,
            predicted_y,
            macro_lagged_for_model_baseline,
        )
        df_test_predict_favorable = predict_test_OOS(
            test_end_index_favorable,
            df_test_complete_favorable.drop("td", axis=1),
            df_test_predict_favorable,
            model_chosen,
            predicted_y,
            macro_lagged_for_model_favorable,
        )

    if need_it_ == "yes_l1l2":
        df_test_predict_adv = rescale_process_l1l2(
            df_test_predict_adv, need_it_, scaler_td
        )
        df_test_predict_baseline = rescale_process_l1l2(
            df_test_predict_baseline, need_it_, scaler_td
        )
        df_test_predict_favorable = rescale_process_l1l2(
            df_test_predict_favorable, need_it_, scaler_td
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=macro_lagged_for_model_adv.index,
            y=df_test_predict_baseline["td_predict"],
            mode="lines",
            name="td baseline",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_lagged_for_model_adv.index,
            y=df_test_predict_adv["td_predict"],
            mode="lines",
            name="td adverse",
            line=dict(color="black"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_lagged_for_model_adv.index,
            y=df_test_predict_favorable["td_predict"],
            mode="lines",
            name="td favorable",
            line=dict(color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=macro_lagged_for_model_adv.index,
            y=df_test_predict_baseline["td_reel"],
            mode="lines",
            name="td observé",
            line=dict(color="yellow"),
        )
    )
    st.plotly_chart(fig)

    add_df_previsions_to_dict(
        df_test_predict_baseline,
        df_test_predict_adv,
        df_test_predict_favorable,
        macro_baseline,
        model_showed,
        df_previsions_dict,
    )

    st.session_state.previsions_dictionary = df_previsions_dict

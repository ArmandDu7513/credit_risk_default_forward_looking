import pandas as pd
import numpy as np
import statsmodels as sm
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import datetime
from scipy.stats import kendalltau, shapiro, kstest, anderson
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import (
    kpss, adfuller
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, het_goldfeldquandt
)
from statsmodels.api import add_constant, OLS

from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

######################################################################################

###################   Transformation des variables macro   ###########################

######################################################################################


def plot_series_brutes(macro):
    fig = make_subplots(
        rows=3,
        cols=2,
        column_width=[0.5, 0.5],  # set the width ratio of columns
        specs=[
            [{"colspan": 2}, None],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=(
            "Taux de défaut",
            "HICP : inflation",
            "RGDP : Produit intérieur brut",
            "RREP: Real estate prices",
            "UNR : Chômage",
        ),
    )

    # Add traces to each subplot
    fig.add_trace(
        go.Scatter(
            x=macro.index, y=macro["td"], mode="lines", line={"color": "#900C3F"}
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=macro.index, y=macro["HICP"], mode="lines", line={"color": "#4c6474"}
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=macro.index, y=macro["RGDP"], mode="lines", line={"color": "#4c6474"}
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=macro.index, y=macro["RREP"], mode="lines", line={"color": "#4c6474"}
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=macro.index, y=macro["UNR"], mode="lines", line={"color": "#4c6474"}
        ),
        row=3,
        col=2,
    )

    # Set axis labels
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Taux (%)", row=1, col=1)
    fig.update_yaxes(title_text="Indice base 100", row=2, col=1)
    fig.update_yaxes(title_text="En €", row=2, col=2)
    fig.update_yaxes(title_text="Indice base 100", row=3, col=1)
    fig.update_yaxes(title_text="Taux (%)", row=3, col=2)

    # Set axis tick label font size
    fig.update_layout(xaxis_tickfont_size=8, yaxis_tickfont_size=8)

    # Adjust the spacing between subplots
    fig.update_layout(
        height=750,
        width=750,
    )
    fig.update_layout(showlegend=False)

    # Show the plot
    st.plotly_chart(fig)


def remove_season(df, list_col):
    """On a vu dans le graphique que des séries peuvent avoir des saisonnalités, alors même que le Taux de défaut n'en a jamais.
    On voit en particulier le HICP qui a une forte saisonnalité. Celle-ci risque de
    fausser les résultats pour la suite. On peut choisir de la retirer des saéries
    """
    for col in list_col:
        decomposition = sm.tsa.seasonal.seasonal_decompose(df[col], period=4)
        seasonal = decomposition.seasonal
        df[col] = df[col] - seasonal
    return df


# Lissage par moyenne mobile des variables indiquées dans la liste 'columns'
def lissage_moyenne_mobile(df, list_col, nb_jours_moyennées=6, fenetre_des_moyennes=3):
    for col in list_col:
        last_observations = df.iloc[:-nb_jours_moyennées][col].tail(
            fenetre_des_moyennes
        )
        series_smoothed = df[col].copy()
        series_smoothed.iloc[-nb_jours_moyennées:] = last_observations.mean()

        df[col] = series_smoothed


def logit(p):
    return np.log(p) - np.log(1 - p)


def import_and_series_transformation(
    drop_na, list_col_saison, list_col_lissage, logit_choix
):
    """
    drop_na : retirer les lignes avec des valeurs manquantes dans la colonne du taux de défaut
    list_col_chosen : les variables pour lesquelles on souhaite estimer puis retirer la saisonnalité
    list_col_lissage : lissage des dernieres observation des variables, par la moyenne mobile
    logit : application oui ou non de la transformation Logit sur le taux de défaut

    """

    # importation des données
    td = pd.read_excel("data/default_rate_quarterly.xlsx")
    macro = pd.read_excel("data/variables_macroeconomiques.xlsx")

    # création du dataframe, avec les variables macro et le TD, index par la date
    td = td.set_index("Date")
    macro = macro.set_index("Date")
    macro["td"] = td

    # retirons la saisonnalités des variables choisies
    macro = remove_season(macro, list_col_saison)

    # Lissage par moyenne mobile des variables indiquées dans la liste 'columns'
    window_size = 10
    lissage_moyenne_mobile(macro, list_col_lissage)

    # Variation mensuelle
    macro[["RGDP_tri", "HICP_tri", "IRLT_tri", "RREP_tri", "UNR_tri"]] = (
        macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]]
        - macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(1)
    ) / macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(1)

    # Variation annuelle
    macro[["RGDP_ann", "HICP_ann", "IRLT_ann", "RREP_ann", "UNR_ann"]] = (
        macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]]
        - macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(4)
    ) / macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(4)

    # retirons les variables brutes
    col_brutes = ["RGDP", "HICP", "IRLT", "RREP", "UNR"]
    macro = macro.drop(col_brutes, axis=1)

    # appliquer le logit sur le taux de défaut ou pas
    if logit_choix == "Oui":
        macro["DD"] = macro["td"].apply(logit)
        macro["td"] = macro["DD"]
        macro = macro.drop("DD", axis=1)

    # drop les valeurs manquantes (selon la colonne du TD)
    if drop_na == "Oui":
        macro.dropna(subset=["td"], how="all", inplace=True)

    return macro


def import_and_series_transformation_and_graph(
    drop_na, list_col_saison, list_col_lissage, logit_choix, variable_graph
):
    """
    drop_na : retirer les lignes avec des valeurs manquantes dans la colonne du taux de défaut
    list_col_chosen : les variables pour lesquelles on souhaite estimer puis retirer la saisonnalité
    list_col_lissage : lissage des dernieres observation des variables, par la moyenne mobile
    logit : application oui ou non de la transformation Logit sur le taux de défaut

    """

    # importation des données
    td = pd.read_excel("data/default_rate_quarterly.xlsx")
    macro = pd.read_excel("data/variables_macroeconomiques.xlsx")

    # création du dataframe, avec les variables macro et le TD, index par la date
    td = td.set_index("Date")
    macro = macro.set_index("Date")
    macro["td"] = td

    # retirons la saisonnalités des variables choisies
    macro = remove_season(macro, list_col_saison)

    # Variation mensuelle
    macro[["RGDP_tri", "HICP_tri", "IRLT_tri", "RREP_tri", "UNR_tri"]] = (
        macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]]
        - macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(1)
    ) / macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(1)

    # Variation annuelle
    macro[["RGDP_ann", "HICP_ann", "IRLT_ann", "RREP_ann", "UNR_ann"]] = (
        macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]]
        - macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(4)
    ) / macro[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(4)

    # retirons les variables brutes
    col_brutes = ["RGDP", "HICP", "IRLT", "RREP", "UNR"]
    macro = macro.drop(col_brutes, axis=1)

    # Lissage par moyenne mobile des variables indiquées dans la liste 'columns'
    window_size = 10
    lissage_moyenne_mobile(macro, list_col_lissage)

    # drop les valeurs manquantes (selon la colonne du TD)
    if drop_na == "Oui":
        macro.dropna(subset=["td"], how="all", inplace=True)

    # appliquer le logit sur le taux de défaut ou pas
    if logit_choix == "Oui":
        macro["DD"] = macro["td"].apply(logit)
        macro["td"] = macro["DD"]
        macro = macro.drop("DD", axis=1)

    font_size = 16
    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create two traces using go.Scatter
    trace1 = go.Scatter(
        x=macro.index, y=macro["td"], name="DR", line=dict(color="#900C3F")
    )
    trace2 = go.Scatter(
        x=macro.index,
        y=macro[variable_graph],
        name=variable_graph,
        line=dict(color="#4c6474"),
    )

    # Add traces to subplot grid, setting secondary_y=True for the second trace
    subfig.add_trace(trace1, secondary_y=False)
    subfig.add_trace(trace2, secondary_y=True)

    # Customize axis labels and titles
    subfig.update_xaxes(title_text="Date")
    subfig.update_yaxes(title_text="DR", secondary_y=False)
    subfig.update_yaxes(title_text=variable_graph, secondary_y=True)

    # Customize layout
    subfig.update_layout(
        title="Evolution de " + variable_graph + "  avec le taux de défaut",
        legend=dict(x=1, y=1, font=dict(size=font_size)),
        height=400,
        width=500,
    )

    # Display plot using st.plotly_chart

    return subfig, macro


def drop_trimestrielle_or_annuelle_if_wanted(df, choix):
    """
    What kind of variables do you wish to keep :
    choix = {'both', 'Trimestrielles', 'Annuelles'}"""

    if choix == "Trimestrielles":
        df = df.filter(regex="^(td|.*_tri)$")
    elif choix == "Annuelles":
        df = df.filter(regex="^(td|.*_ann)$")
    else:
        pass
    return df


def list_str(list_to_change):
    columns = [col.strip("[]'") for col in list_to_change]
    return columns


######################################################################################

####################   Stationnarité des variables macro   ###########################

######################################################################################


def ljung_box(df, drop_td):
    if drop_td == "Yes":
        macro_Box = df.drop("td", axis=1).dropna()
    else:
        macro_Box = df.dropna()
    results = pd.DataFrame(
        columns=["Variable", "Ljung-Box Statistic", "p-value", "Résultat"]
    )
    for col in macro_Box:
        res = ARIMA(macro_Box[col], order=(1, 0, 1)).fit()
        # print(sm.stats.acorr_ljungbox(res.resid, lags=[5]))
        results.loc[len(results)] = [
            col,
            acorr_ljungbox(res.resid, lags=[5]).iloc[0, 0],
            acorr_ljungbox(res.resid, lags=[5]).iloc[0, 1],
            acorr_ljungbox(res.resid, lags=[5]).iloc[0, 1] < 0.05,
        ]
    results["Résultat"] = results["Résultat"].replace(
        False, "Les résidus sont indépendants"
    )
    results["Résultat"] = results["Résultat"].replace(
        True,
        "Les résidus NE sont PAS indépendants : il y a de l'autocorrélation dans la série",
    )

    return results


def show_acf_pacf(df, variable):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # plot ACF on the left subplot
    plot_acf(df.dropna()[variable], lags=8, ax=axs[0])
    axs[0].set_title("ACF of " + variable)

    # plot PACF on the right subplot
    plot_pacf(df.dropna()[variable], lags=8, ax=axs[1])
    axs[1].set_title("PACF of " + variable)

    # show the plot for each variable
    st.pyplot(fig)


def kpss_test(df, regression, alpha="5%"):
    # Initialisation du DataFrame de résultats
    results = pd.DataFrame(
        columns=[
            "Test",
            "Variable",
            "Regression",
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Résultat",
        ]
    )

    # Boucle sur chaque colonne de la DataFrame
    for col in df.columns:
        # Calcul du test KPSS pour la série temporelle
        kpss_result = kpss(df[col], regression=regression)

        # Ajout des résultats du test ADF dans le DataFrame de résultats
        results.loc[len(results)] = [
            "KPSS",
            col,
            regression,
            round(kpss_result[0], 3),
            round(kpss_result[1], 3),
            kpss_result[2],
            kpss_result[0] < kpss_result[3][alpha],
        ]

    # Remplacement des valeurs booléennes par des chaînes de caractères lisibles
    results["Résultat"] = results["Résultat"].replace(False, "Non Stationnaire")
    results["Résultat"] = results["Résultat"].replace(True, "Stationnaire")

    return results


def PhillipsPerron_test(df, regression, alpha="5%"):
    # Initialisation du DataFrame de résultats
    results = pd.DataFrame(
        columns=[
            "Test",
            "Variable",
            "Regression",
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Résultat",
        ]
    )

    # Boucle sur chaque colonne de la DataFrame
    for col in df.columns:
        # Calcul du test ADF pour la série temporelle
        pp_result = PhillipsPerron(df[col], trend=regression)
        # Ajout des résultats du test ADF dans le DataFrame de résultats
        results.loc[col] = [
            "PP",
            col,
            regression,
            round(pp_result.stat, 3),
            round(pp_result.pvalue, 3),
            pp_result.lags,
            pp_result.pvalue < 0.05,
        ]

    # Remplacement des valeurs booléennes par des chaînes de caractères lisibles
    results["Résultat"] = results["Résultat"].replace(False, "Non Stationnaire")
    results["Résultat"] = results["Résultat"].replace(True, "Stationnaire")

    return results


def AugmenteDickeyFuller_test(df, regression, alpha="5%"):
    # Initialisation du DataFrame de résultats
    results = pd.DataFrame(
        columns=[
            "Test",
            "Variable",
            "Regression",
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Résultat",
        ]
    )

    # Boucle sur chaque colonne de la DataFrame
    for col in df.columns:
        # Calcul du test ADF pour la série temporelle
        adf_result = adfuller(df[col], regression=regression)
        # Ajout des résultats du test ADF dans le DataFrame de résultats
        results.loc[col] = [
            "ADF",
            col,
            regression,
            round(adf_result[0], 3),
            round(adf_result[1], 3),
            adf_result[2],
            adf_result[1] < 0.05,
        ]

    # Remplacement des valeurs booléennes par des chaînes de caractères lisibles
    results["Résultat"] = results["Résultat"].replace(False, "Non Stationnaire")
    results["Résultat"] = results["Résultat"].replace(True, "Stationnaire")

    return results


def highlight_variables(row):
    if row["Variable"] == "HICP_ann":
        return ["background-color: linen"] * len(row)
    if row["Variable"] == "HICP_tri":
        return ["background-color: peru"] * len(row)
    if row["Variable"] == "IRLT_ann":
        return ["background-color: seashell"] * len(row)
    if row["Variable"] == "IRLT_tri":
        return ["background-color: palegreen"] * len(row)
    if row["Variable"] == "RGDP_ann":
        return ["background-color: azure"] * len(row)
    if row["Variable"] == "RGDP_tri":
        return ["background-color: lightblue"] * len(row)
    if row["Variable"] == "RREP_ann":
        return ["background-color: lavender"] * len(row)
    if row["Variable"] == "RREP_tri":
        return ["background-color: thistle"] * len(row)
    if row["Variable"] == "UNR_ann":
        return ["background-color: lavenderbush"] * len(row)
    if row["Variable"] == "UNR_tri":
        return ["background-color: lightpink"] * len(row)
    else:
        return ["background-color: chocolate"] * len(row)


def highlight_value(val):
    if val == "Les résidus sont indépendants":
        return "color: white; background-color: green"
    else:
        return "color: white; background-color: red"


def highlight_result(val):
    if val == "Stationnaire":
        return "color: white; background-color: green"
    else:
        return "color: white; background-color: red"


def concat_tests_and_sort(kpss, pp, adf):
    results_of_tests = pd.concat([kpss, pp, adf], ignore_index=True)
    results_of_tests = results_of_tests.drop("Test Statistic", axis=1)
    results_of_tests = results_of_tests.sort_values(
        ["Variable", "Regression"], ascending=[True, False]
    )
    return results_of_tests


def draw_conclusion_stationary(results_all_tests, nb_variables_studies):
    # Group the dataframe by 'Variable' and 'Résultat', then get the size of each group
    result_counts = (
        results_all_tests.groupby(["Variable", "Résultat"])
        .size()
        .reset_index(name="Count")
    )

    # Filter the groups where 'Résultat' is 'Stationary'
    result_counts = result_counts[result_counts["Résultat"] == "Stationnaire"]

    # Drop the 'Résultat' column from the result_counts dataframe
    result_counts = result_counts.drop("Résultat", axis=1)

    # Rename the 'Count' column to 'Stationary_count'
    result_counts = result_counts.rename(columns={"Count": "Stationary_count"})

    # Merge the result_counts dataframe with the original results_all_tests dataframe
    results_all_tests = results_all_tests.merge(
        result_counts, on="Variable", how="left"
    )

    # Group the dataframe by 'Variable', then get the count of rows for each group
    row_counts = results_all_tests.groupby("Variable").size()

    # Divide the 'Stationary_count' column by the row counts for each variable
    results_all_tests["Stationary_ratio"] = results_all_tests["Stationary_count"] / (
        len(results_all_tests) / nb_variables_studies
    )

    # Fill NaN values with 0
    results_all_tests["Stationary_ratio"] = results_all_tests[
        "Stationary_ratio"
    ].fillna(0)

    results_all_tests["Proportion de conclusion positive pour la stationnarité (%)"] = (
        round(results_all_tests["Stationary_ratio"], 4) * 100
    )

    # decision finale sur la stationnarité de la variable
    results_all_tests.loc[
        results_all_tests["Proportion de conclusion positive pour la stationnarité (%)"]
        > 50,
        "Décision Finale",
    ] = "Stationnaire"
    results_all_tests.loc[
        results_all_tests["Proportion de conclusion positive pour la stationnarité (%)"]
        <= 50,
        "Décision Finale",
    ] = "Non Stationnaire"

    return results_all_tests


def readable_conclusions(results_all_tests):
    # Drop non useful columns
    col_drop = [
        "Test",
        "p-value",
        "Regression",
        "Regression",
        "Lags Used",
        "Résultat",
        "Stationary_count",
        "Stationary_ratio",
    ]
    results_all_tests = results_all_tests.drop(col_drop, axis=1)

    # Drop duplicate rows
    results_all_tests = results_all_tests.drop_duplicates()

    # Reset the index
    results_all_tests = results_all_tests.reset_index(drop=True)

    return results_all_tests


######################################################################################

#######################   Stationnarité du taux de défaut   ##########################

######################################################################################


def newey_west_tests_find_beta(dataframe, test):
    """On doit faire le test de Newey-West, en utilisant la série DRt - Beta*DRt-1. On passe par les tests KPSS, PP et ADF"""

    # creation du dataframe qu'on utilisera à chaque fois
    df = pd.DataFrame()
    df["td"] = dataframe["td"]
    df["td_minus1"] = df["td"].shift(1)
    df = df.dropna()

    # create an empty DataFrame for the results with columns for betax, stat, crit, and regression type
    df_results = pd.DataFrame(columns=["betax", "stat", "crit", "regression type"])
    fig = go.Figure()

    betax = []
    pvalues = []
    crit = []
    stat = []

    for beta in range(100):
        df["DRt_beta"] = (
            df["td"] - (beta / 100) * df["td_minus1"]
        )  # creation de la série qu'on va utiliser pour mener le test

        if test == "KPSS":
            reg = "c"
            kpss_result = kpss(df["DRt_beta"], regression=reg)
            stat.append(kpss_result[0])
            pvalues.append(kpss_result[1])
            crit.append(kpss_result[3]["5%"])  # récupérer la valeur critique
        elif test == "PP":
            reg = "c"
            pp_result = PhillipsPerron(df["DRt_beta"], trend=reg)
            stat.append(pp_result.stat)
            pvalues.append(pp_result.pvalue)
            crit.append(pp_result.critical_values["5%"])  # récupérer la valeur critique
        else:
            reg = "c"
            adf_result = adfuller(df["DRt_beta"], regression=reg)
            stat.append(adf_result[0])
            pvalues.append(adf_result[1])
            crit.append(adf_result[4]["5%"])  # récupérer la valeur critique
        df = df.drop("DRt_beta", axis=1)
        betax.append(beta / 100)

    # create a new DataFrame for the current regression type --> utile dans le cas où je mets plusieurs types de regression à tester
    df_results = pd.DataFrame(
        {"betax": betax, "stat": stat, "crit": crit, "regression type": [reg] * 100}
    )

    # add the results to the main DataFrame

    # fig.add_trace(go.Scatter(x=betax, y=stat, name=test + " Statistics", line=dict(color='mediumvioletred')))
    # fig.add_trace(go.Scatter(x=betax, y=crit, name="valeurs critiques", line=dict(color='crimson')))
    fig.add_trace(
        go.Scatter(
            x=betax,
            y=[0.05] * np.shape(crit)[0],
            name="Seuil de 5%",
            line=dict(color="darkslategrey"),
        )
    )
    fig.add_trace(
        go.Scatter(x=betax, y=pvalues, name="pvalue", line=dict(color="skyblue"))
    )
    fig.update_xaxes(title_text="Beta values")
    fig.update_yaxes(title_text="p-values")

    fig.update_layout(
        title="Valeurs de Beta selon le test " + test,
        font_size=7,
        height=250,
        width=300,
    )
    st.plotly_chart(fig)
    st.caption("Test statistique Newey & West sur le Beta")
    return df_results


def AAAAnewey_west_tests_find_beta(dataframe, test):
    """On doit faire le test de Newey-West, en utilisant la série DRt - Beta*DRt-1. On passe par les tests KPSS, PP et ADF"""

    # creation du dataframe qu'on utilisera à chaque fois
    df = pd.DataFrame()
    df["td"] = dataframe["td"]
    df["td_minus1"] = df["td"].shift(1)
    df = df.dropna()

    # create an empty DataFrame for the results with columns for betax, stat, crit, and regression type
    df_results = pd.DataFrame(columns=["betax", "stat", "crit", "regression type"])
    fig, ax = plt.subplots()
    betax = []
    pvalues = []
    crit = []
    stat = []

    for beta in range(100):
        df["DRt_beta"] = (
            df["td"] - (beta / 100) * df["td_minus1"]
        )  # creation de la série qu'on va utiliser pour mener le test

        if test == "KPSS":
            reg = "c"
            kpss_result = kpss(df["DRt_beta"], regression=reg)
            stat.append(kpss_result[0])
            pvalues.append(kpss_result[1])
            crit.append(kpss_result[3]["5%"])  # récupérer la valeur critique
        elif test == "PP":
            reg = "c"
            pp_result = PhillipsPerron(df["DRt_beta"], trend=reg)
            stat.append(pp_result.stat)
            pvalues.append(pp_result.pvalue)
            crit.append(pp_result.critical_values["5%"])  # récupérer la valeur critique
        else:
            reg = "c"
            adf_result = adfuller(df["DRt_beta"], regression=reg)
            stat.append(adf_result[0])
            pvalues.append(adf_result[1])
            crit.append(adf_result[4]["5%"])  # récupérer la valeur critique
        df = df.drop("DRt_beta", axis=1)
        betax.append(beta / 100)

    # create a new DataFrame for the current regression type --> utile dans le cas où je mets plusieurs types de regression à tester
    df_results = pd.DataFrame(
        {"betax": betax, "stat": stat, "crit": crit, "regression type": [reg] * 100}
    )
    # add the results to the main DataFrame
    # df_results = pd.concat([df_results, results_df_temporary], ignore_index=True)

    ax.plot(betax, stat, color="mediumvioletred", label=test + " Statistics")
    ax.plot(betax, crit, color="crimson", label="valeurs critiques")
    ax.plot(
        betax, [0.05] * np.shape(crit)[0], color="darkslategrey", label="Seuil de 5%"
    )
    ax.plot(betax, pvalues, color="skyblue", label="pvalue")
    plt.title("Valeurs de Beta selon le test " + test)

    plt.legend()
    plt.show()
    plt.xlabel("Valeurs de Beta")
    plt.ylabel("Test statistique Newey & West sur le Beta")
    plt.figure()
    st.pyplot(fig)
    return df_results


######################################################################################

###############################   La Corrélation   ###################################

######################################################################################


def compute_var_lagged(macro_lags, nb_lag_var_ann, nb_lag_var_tri, nb_lag_var_td):
    macro_corr = macro_lags.copy()
    for col in macro_corr.filter(regex="_ann").columns:
        # calculer les Lags
        for lag_number in range(1, nb_lag_var_ann + 1):
            macro_corr[col + "_lag" + str(lag_number)] = macro_corr[col].shift(
                lag_number
            )

    for col in macro_corr.filter(regex="_tri").columns:
        # calculer les Lags
        for lag_number in range(1, nb_lag_var_tri + 1):
            macro_corr[col + "_lag" + str(lag_number)] = macro_corr[col].shift(
                lag_number
            )
    for col in macro_corr.filter(regex="td").columns:
        for lag_number in range(1, nb_lag_var_td + 1):
            macro_corr[col + "_lag" + str(lag_number)] = macro_corr[col].shift(
                lag_number
            )
    return macro_corr


def commpute_corr_matrix(macro_lagged):
    corr_df = pd.DataFrame(index=macro_lagged.columns, columns=macro_lagged.columns)
    macro_lagged_no_nan = macro_lagged.dropna()
    for col1 in macro_lagged_no_nan.columns:
        for col2 in macro_lagged_no_nan.columns:
            corr, _ = kendalltau(macro_lagged_no_nan[col1], macro_lagged_no_nan[col2])
            corr_df.loc[col1, col2] = corr
    return corr_df


def choosing_seuil_corr_to_DR(df, seuil):
    """on choisit le seuil de correlation qu'on veut, puis on crée une liste avec le nom des variables qu'on garde"""

    df_var_VERY_corr_to_DR = df[df["Kendall Correlations"] > seuil]
    list_var_in_model = df_var_VERY_corr_to_DR.index.tolist()
    return list_var_in_model


def commpute_corr_matrix_var_macro(macro_lagged, list_var_over_threshold):
    macro_lagged = macro_lagged.dropna()
    corr_df_macro = pd.DataFrame(
        index=list_var_over_threshold, columns=list_var_over_threshold
    )
    for col1 in macro_lagged[list_var_over_threshold]:
        for col2 in macro_lagged[list_var_over_threshold]:
            corr, _ = kendalltau(
                macro_lagged[list_var_over_threshold][col1],
                macro_lagged[list_var_over_threshold][col2],
            )
            corr_df_macro.loc[col1, col2] = corr
    return corr_df_macro


def find_high_corr_pairs(corr_matrix, threshold):
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    return high_corr_pairs


######################################################################################

##############################   La Modélisation   ###################################

######################################################################################
#                                                                                    #
##############################   Train Test Split  ###################################


def split_train_test(df):
    df.index = pd.to_datetime(df.index)

    # Get index object
    idx = df.index

    # Define time interval for train and test sets
    train_start_date = datetime.date(2010, 7, 31)
    train_end_date = datetime.date(2017, 10, 31)
    test_start_date = datetime.date(2017, 10, 31)
    test_end_date = datetime.date(2019, 10, 31)

    # Split data into train and test sets based on time interval
    train_data = df[(idx.date >= train_start_date) & (idx.date <= train_end_date)]
    test_data = df[(idx.date > test_start_date) & (idx.date <= test_end_date)]

    # Display train and test set sizes
    print("Train set size:", len(train_data))
    print("Test set size:", len(test_data))

    return train_data, test_data


#                                                                                    #
#########################    Coeur de la modélisation      ###########################


def model_OLS(df_lags, varaibles_explicatives_allowed):
    y = df_lags["td"]
    X = df_lags[varaibles_explicatives_allowed]
    # Sort columns alphabetically
    sorted_columns = sorted(X.columns)
    # Reindex DataFrame using sorted column names
    X = X[sorted_columns]
    X = add_constant(X)
    df_var = X
    results = OLS(y, X).fit()  # toujours mettre une constante dans le modèle

    # Verify the assumptions of normality and multicollinearity
    resid = results.resid
    p_shapiro = shapiro(resid).pvalue
    p_kolmogorov = kstest(resid, "norm").pvalue
    boolean_anderson = anderson(resid).statistic > anderson(resid).critical_values[2]
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
    nb_vif_high = (vif["VIF"] > 5).sum()

    # Save the results in the DataFrame
    params_dict = {
        var + "_param": param
        for var, param in zip(results.params.index, results.params)
    }
    # params_dict.pop("const_param", None) #Je veux garder la constante, et connaitre sa valeur
    pvals_dict = {
        var + "_pval": results.pvalues[i] for i, var in enumerate(results.params.index)
    }
    pvals_dict.pop("const_pval", None)
    errors_for_theil = (np.sqrt(sum(np.array(y) ** 2) / len(y))) + (
        np.sqrt(
            sum(np.array(results.predict(add_constant(X))) ** 2)
            / len(results.predict(add_constant(X)))
        )
    )

    row = {
        "variables": [results.params.index],
        "bic": results.bic,
        "logv": results.llf,
        "R2": results.rsquared,
        "R2_score": r2_score(y, results.predict()),
        "R2_aj": results.rsquared_adj,
        "RMSE": np.sqrt(results.mse_resid),
        "U de Theil": np.sqrt(results.mse_resid) / errors_for_theil,
        "pval_shapiro": p_shapiro,
        "pval_kolmogorov": p_kolmogorov,
        "boolean_anderson": boolean_anderson,
        "mean_resid": resid.mean(),
        "nb_var_vif_above_5": nb_vif_high,
        "nb_vars": X.shape[1] - 1,
        **params_dict,
        **pvals_dict,
    }

    # On return un dataframe avec les informations clés du modèle, le modèle lui-même sous forme OLS().fit, la variable cible le dataframe créé de variables explicatives
    return row, results, y, df_var


def train_and_evaluate_rf(
    n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """C'est une fonction qui permet de faire tourner un RF, tout en calculant ces métriques sur les échantillons train et test pour les comparer, pour véreifier qu'il n'y a pas d'overfit"""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    train_rmse = round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4)
    test_rmse = round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4)
    return train_rmse, test_rmse


def predict_BMA(model_saved, df, df_row_parameters_qui_stock_le_modele):
    # Get the column names of df_X and df_row_parameters_qui_stock_le_modele
    X_cols = set(df.columns)
    params_cols = set(df_row_parameters_qui_stock_le_modele.columns)
    # Attention : on perd la constance ici, ON DOIT LA RAJOUTER ENSUITE

    # Find the common columns between the two sets
    common_cols = list(X_cols.intersection(params_cols))

    # Convert the DataFrames to numpy arrays --> with only needed columns
    X_observed = df[common_cols].values
    params = df_row_parameters_qui_stock_le_modele[common_cols].values.reshape(-1, 1)

    # Multiply X and params using matrix multiplication
    pred_values = X_observed @ params

    # Adding the intercept...
    pred_values_BMA = pred_values + df_row_parameters_qui_stock_le_modele.const[0]

    return pred_values_BMA


#                                                                                    #
#########################    Checking training period      ###########################


def graphique_periode_entrainement(
    df, sentence, cible, predict, streamlit_or_no, df_lag_for_index
):
    df_no_nan_td = df.dropna()
    RMSE_model = round(
        np.sqrt(
            mean_squared_error(
                df_no_nan_td[cible].to_numpy(), df_no_nan_td[predict].to_numpy()
            )
        ),
        4,
    )
    # Create a line plot of 'td' vs 'y_pred'
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_lag_for_index.index,
            y=df[cible],
            mode="lines",
            name="td",
            line=dict(color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_lag_for_index.index,
            y=df[predict],
            mode="lines",
            name="td_predicted",
            line=dict(color="blue"),
        )
    )

    # Add axis labels and a title
    fig.update_layout(
        xaxis_title="Dates",
        yaxis_title="Taux (%)",
        title="Graphique du dataset d'entrainement en utilisant " + sentence,
    )
    # Add subtitle
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
    if streamlit_or_no == "streamlit":
        st.plotly_chart(fig)
        # Display the plot
    else:
        fig.show()


def interpretation_tests(df):
    hypothesis = pd.DataFrame(df)[
        [
            "pval_shapiro",
            "pval_kolmogorov",
            "mean_resid",
            "nb_var_vif_above_5",
            "boolean_anderson",
        ]
    ]
    conclusion_tests_normalite = 0
    if hypothesis["mean_resid"][0] < 0.5:
        print(
            "La moyenne des résidus est très proche de zéro. L'hypothèse faible de normalité des résidus est respectée."
        )
    else:
        print(
            "La moyenne des résidus est plutôt loin de zéro. L'hypothèse faible de normalité des résidus n'est pas respectée"
        )
    print("-------")
    if hypothesis["pval_shapiro"][0] > 0.05:
        print(
            "Shapiro Wilk : On ne peut pas dire que les résidus ne suivent pas une loi normale. Donc on accepte l'hypothèse de normalité des résidus"
        )
        conclusion_tests_normalite += 1
    else:
        print(
            "On peut  dire que les résidus ne suivent pas une loi normale. Donc on rejette l'hypothèse de normalité des résidus"
        )

    if hypothesis["pval_kolmogorov"][0] < 0.05:
        print(
            "Kolmogorov Smirov : On rejette l'hypothèse de normalité (two-sided test)."
        )
    else:
        print("On accepte l'hypothèse de normanité des résidus")
        conclusion_tests_normalite += 1
    if hypothesis["boolean_anderson"][0] == True:
        print(
            "Anderson Darling : On rejette l'hypothèse H0 du test, donc on rejette la normalité des résidus"
        )
    else:
        print(
            "On accepte l'hypothèse H0 du test, donc on accepte la normalité des résidus"
        )
        conclusion_tests_normalite += 1
    if conclusion_tests_normalite > 1:
        print(
            "{}% des tests de normalité admettent la normalité des résidus, sur 3 tests.".format(
                round(conclusion_tests_normalite / 3 * 100), 0
            )
        )
    print("-------")
    print(
        "Le modèle admet {} variables avec un facteur VIF supérieur à 5".format(
            hypothesis["nb_var_vif_above_5"][0]
        )
    )


def interpretation_tests_st(df):
    hypothesis = pd.DataFrame(df)[
        [
            "pval_shapiro",
            "pval_kolmogorov",
            "mean_resid",
            "nb_var_vif_above_5",
            "boolean_anderson",
        ]
    ]
    conclusion_tests_normalite = 0
    if hypothesis["mean_resid"][0] < 0.5:
        st.write(
            "La moyenne des résidus est très proche de zéro. L'hypothèse faible de normalité des résidus est respectée."
        )
    else:
        st.write(
            "La moyenne des résidus est plutôt loin de zéro. L'hypothèse faible de normalité des résidus n'est pas respectée"
        )
    st.write("-------")
    if hypothesis["pval_shapiro"][0] > 0.05:
        st.write(
            "Shapiro Wilk : On ne peut pas dire que les résidus ne suivent pas une loi normale. Donc on accepte l'hypothèse de normalité des résidus"
        )
        conclusion_tests_normalite += 1
    else:
        st.write(
            "On peut dire que les résidus ne suivent pas une loi normale. Donc on rejette l'hypothèse de normalité des résidus"
        )

    if hypothesis["pval_kolmogorov"][0] < 0.05:
        st.write(
            "Kolmogorov Smirov : On rejette l'hypothèse de normalité (two-sided test)."
        )
    else:
        st.write("Kolmogorov Smirov : On accepte l'hypothèse de normalité des résidus")
        conclusion_tests_normalite += 1
    if hypothesis["boolean_anderson"][0] == True:
        st.write(
            "Anderson Darling : On rejette l'hypothèse H0 du test, donc on rejette la normalité des résidus"
        )
    else:
        st.write(
            "Anderson Darling : On accepte l'hypothèse H0 du test, donc on accepte la normalité des résidus"
        )
        conclusion_tests_normalite += 1
    if conclusion_tests_normalite > 1:
        st.write(
            "{}% des tests de normalité admettent la normalité des résidus, sur 3 tests.".format(
                round(conclusion_tests_normalite / 3 * 100), 0
            )
        )
    st.write("-------")


def tests_homoscedasticide(res, endog_var, exog_var):
    pval_breuschpagan = het_breuschpagan(resid=res, exog_het=exog_var)[1]
    pval_white = het_white(resid=res, exog=exog_var)[1]
    pval_goldfeldquandt = het_goldfeldquandt(y=endog_var, x=exog_var)[1]
    print("Pvalue du test Breusch Pagan : ", str(pval_breuschpagan))
    print("Pvalue du test White : ", str(pval_white))
    print("Pvalue du test Goldfeldquandt : ", str(pval_goldfeldquandt))


def tests_homoscedasticide_st(res, endog_var, exog_var):
    pval_breuschpagan = het_breuschpagan(resid=res, exog_het=exog_var)[1]
    pval_white = het_white(resid=res, exog=exog_var)[1]

    if pval_breuschpagan < 0.05:
        st.write(
            "Breush Pagan : On doit rejeter H0 : il y a de l'hétéroscédasticité dans les résidus"
        )
    else:
        st.write(
            "Breush Pagan : On accepte H0, donc il n'y a pas d'hétéroscédasticité."
        )
    if pval_white < 0.05:
        st.write(
            "White : On doit rejeter H0 : il y a de l'hétéroscédasticité dans les résidus"
        )
    else:
        st.write("White : On accepte H0, donc il n'y a pas d'hétéroscédasticité.")


#                                                                                    #
##########################    Checking testing period      ###########################


def apply_same_preprocess_as_OLS(X_test, variables_kept_significatives):
    X_test = X_test[variables_kept_significatives]
    sorted_columns = sorted(X_test.columns)
    X_test = X_test[sorted_columns]
    X_test = add_constant(X_test)

    return X_test


# suffit de changer la variable : results_signi pour recréer les prédiction pour différents modèles entraînées
def predict_test(
    index_ending_train,
    df_test_to_fill_in,
    df_test_predicted,
    results_model,
    td_init,
    df_du_debut,
):
    for i in range(index_ending_train, len(df_test_to_fill_in)):
        # Add predicted value for previous date to predictors
        try:
            df_test_complete_i = df_test_to_fill_in.iloc[[i]]

        except:
            df_test_to_fill_in.loc[i, "td_lag1"] = td_init
            df_test_complete_i = df_test_to_fill_in.iloc[[i]]

        # Make prediction for current date
        predicted_y_i = results_model.predict(df_test_complete_i)

        df_test_predicted.loc[df_test_to_fill_in.index[i], "td_predict"] = td_init

        # Update predicted value for next date
        try:
            td_init = predicted_y_i.values[0]
        except:
            td_init = predicted_y_i
    df_lags_td = df_du_debut["td"].reset_index(
        drop=True
    )  # utilisation de la colonne avec tous les DR connus...
    df_test_predicted[
        "td_reel"
    ] = df_lags_td  # pour ensuite les comparer dans le dataframe avec td-estimé.

    return df_test_predicted


def predict_test_OOS(
    index_ending_train,
    df_test_to_fill_in,
    df_test_predicted,
    results_model,
    td_init,
    df_du_debut,
):
    for i in range(index_ending_train, len(df_test_to_fill_in)):
        # Add predicted value for previous date to predictors
        df_test_to_fill_in.loc[i, "td_lag1"] = td_init
        df_test_complete_i = df_test_to_fill_in.iloc[[i]]

        # Make prediction for current date
        predicted_y_i = results_model.predict(df_test_complete_i)

        df_test_predicted.loc[df_test_to_fill_in.index[i], "td_predict"] = td_init

        # Update predicted value for next date
        try:
            td_init = predicted_y_i.values[0]
        except:
            td_init = predicted_y_i
    df_lags_td = df_du_debut["td"].reset_index(
        drop=True
    )  # utilisation de la colonne avec tous les DR connus...
    df_test_predicted[
        "td_reel"
    ] = df_lags_td  # pour ensuite les comparer dans le dataframe avec td-estimé.

    return df_test_predicted


#                                                                                    #
###############    Prédire sur les échantillons Out of Sample      ###################


def prepare_df_for_prediction(df, end_date):
    df_test_complete = df.copy()
    test_end_index = df_test_complete.index.get_loc(
        pd.to_datetime(end_date)
    )  # je vais prédire à partir de la fin de l'entrainement : 2019 Q4
    df_test_complete = df_test_complete.reset_index(drop=True)
    df_test_predict = df_test_complete.copy()

    return df_test_complete, test_end_index, df_test_predict


##################################################################################################
###############    APPLICATION DES SCENARIOS GRACE AUX MODELES PRECEDANTS      ###################


def preprocessing_global_for_prediction(df, df_col_keep):
    # preprocessing qu'on applique pour tous les modèles

    # Variation mensuelle
    df[["RGDP_tri", "HICP_tri", "IRLT_tri", "RREP_tri", "UNR_tri"]] = (
        df[["RGDP", "HICP", "IRLT", "RREP", "UNR"]]
        - df[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(1)
    ) / df[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(1)
    # Variation annuelle
    df[["RGDP_ann", "HICP_ann", "IRLT_ann", "RREP_ann", "UNR_ann"]] = (
        df[["RGDP", "HICP", "IRLT", "RREP", "UNR"]]
        - df[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(4)
    ) / df[["RGDP", "HICP", "IRLT", "RREP", "UNR"]].shift(4)
    # retirons les variables brutes
    col_brutes = ["RGDP", "HICP", "IRLT", "RREP", "UNR"]
    df = df.drop(col_brutes, axis=1)

    macro_lagged_adv = compute_var_lagged(df, 2, 5, 1)
    macro_lagged_for_model = macro_lagged_adv.copy()
    cols_to_keep = ["td", "td_lag1"]
    cols_to_drop = [
        col for col in macro_lagged_for_model.columns if col not in cols_to_keep
    ]
    macro_lagged_for_model.dropna(subset=cols_to_drop, inplace=True)

    macro_lagged_for_model = macro_lagged_for_model[df_col_keep.columns]

    return macro_lagged_for_model


def preprocessing_additionnal_for_OLS_for_prediction(df, list_col_exclude_signi):
    # preprocessing seulement utile et necessaire pour le modèle OLS

    # retirer les mêmes variables que les variables non significatives : bien penser au contraire à garder les variables utilisées pour entraîner le modèle
    variables_kept_signi = list(df.columns)
    for variable in list_col_exclude_signi:
        variables_kept_signi.remove(variable)
    y = df["td"]
    X = df[variables_kept_signi]
    # Sort columns alphabetically and Reindex DataFrame using sorted column names
    sorted_columns = sorted(X.columns)
    X = X[sorted_columns]
    X = add_constant(X)
    return X


def preprocess_OLS_sort_exclude(
    df, yes_or_no, variables_excluded_signi_attention_pas_meme_nom_sur_st
):
    # besoin d'exclure du dataframe de prediction les memes variables que celles retirées pendant l'entrainement du modèle

    if yes_or_no == "yes_ols":
        macro_lagged_for_model_adv_avec_y = df.copy()
        df = preprocessing_additionnal_for_OLS_for_prediction(
            df, variables_excluded_signi_attention_pas_meme_nom_sur_st
        )
        df["td"] = macro_lagged_for_model_adv_avec_y["td"].values
    return df


def predict_test_BMA_OOS(
    test_end_index,
    df_test_complete,
    df_test_predict,
    df_row_parameters,
    predicted_y,
    df_lags,
):
    # l'enregistrement du modèle BMA n'est pas réalisé comme l'enregistrement des modèles scikit learn : donc on a une fonction spécifique pour prédire

    df_test_complete.drop("td", axis=1, inplace=True)

    for i in range(test_end_index, len(df_test_complete)):
        print(predicted_y)

        df_test_complete.loc[i, "td_lag1"] = float(predicted_y)

        print(df_test_complete.loc[i, "td_lag1"])
        df_test_complete_i = df_test_complete.iloc[[i]]
        # print(predicted_y)
        ################################# PREDICTION ########################################
        # Make prediction for current date
        predicted_y_i = predict_BMA(
            df_row_parameters, df_test_complete_i, df_row_parameters
        )
        # print(float(predicted_y_i))
        ################################# PREDICTION ########################################
        df_test_predict.loc[df_test_complete.index[i], "td_predict"] = predicted_y

        # Update predicted value for next date
        predicted_y = float(predicted_y_i)
        # print(predicted_y)

    df_lags_td = df_lags["td"].reset_index(
        drop=True
    )  # utilisation de la colonne avec tous les DR connus...
    df_test_predict[
        "td_reel"
    ] = df_lags_td  # pour ensuite les comparer dans le dataframe avec td-estimé.

    # df_test_complete.drop('td', axis=1)
    return df_test_predict


def rescale_process_l1l2(df, yes_or_no, scaler_td_issu_du_L1L2):
    # besoin de rescale les donnees en output de la regression elastic net à cause de la normalisation utilisée au depart

    if yes_or_no == "yes_l1l2":
        df_td_predict = pd.DataFrame(
            scaler_td_issu_du_L1L2.inverse_transform(df[["td_predict"]])
        )
        df_td_reel = pd.DataFrame(
            scaler_td_issu_du_L1L2.inverse_transform(df[["td_reel"]])
        )

        elastic_net_results_on_cible = pd.DataFrame()
        elastic_net_results_on_cible["td_predict"] = df_td_predict
        elastic_net_results_on_cible["td_reel"] = df_td_reel
    else:
        elastic_net_results_on_cible = df
    return elastic_net_results_on_cible


def add_df_previsions_to_dict(
    df_predicted_base,
    df_predicted_adv,
    df_predicted_favo,
    df_for_index,
    model_showed,
    dict,
):
    # pour ajouter les scenarios prédits, et garder en memoire les prédictions des autres modèles

    df_previsions = pd.concat(
        [
            df_predicted_base["td_predict"],
            df_predicted_adv["td_predict"],
            df_predicted_favo["td_predict"],
        ],
        axis=1,
    ).tail(12)
    df_previsions.columns = ["central", "adverse", "favorable"]
    df_previsions = df_previsions.set_index(df_for_index.tail(12).index)
    dict["{}".format(model_showed)] = df_previsions

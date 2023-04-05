import streamlit as st
import pandas as pd
import numpy as np
import statsmodels as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import kpss
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import kendalltau


######################################################################################

###################   Transformation des variables macro   ###########################

######################################################################################

def plot_series_brutes(macro):
    
    fig = make_subplots(rows=3, cols=2, 
                        column_width=[0.5, 0.5], # set the width ratio of columns
                        specs=[[{'colspan': 2}, None],
                            [{'type': 'scatter'}, {'type': 'scatter'}],
                            [{'type': 'scatter'}, {'type': 'scatter'}]],
                        subplot_titles=('Taux de défaut', 'HICP : inflation', 'RGDP : Produit intérieur brut', 'RREP: ???????', 'UNR : Chômage'))

    # Add traces to each subplot
    fig.add_trace(go.Scatter(x=macro.index, y=macro['td'], mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=macro.index, y=macro['HICP'], mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=macro.index, y=macro['RGDP'], mode='lines'), row=2, col=2)
    fig.add_trace(go.Scatter(x=macro.index, y=macro['RREP'], mode='lines'), row=3, col=1)
    fig.add_trace(go.Scatter(x=macro.index, y=macro['UNR'], mode='lines'), row=3, col=2)

    # Set axis labels
    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_yaxes(title_text='Taux (%)', row=1, col=1)
    fig.update_yaxes(title_text='Indice base 100', row=2, col=1)
    fig.update_yaxes(title_text='En €', row=2, col=2)
    fig.update_yaxes(title_text='Indice base 100', row=3, col=1)
    fig.update_yaxes(title_text='Taux (%)', row=3, col=2)

    # Set axis tick label font size
    fig.update_layout(xaxis_tickfont_size=8, yaxis_tickfont_size=8)

    # Adjust the spacing between subplots
    fig.update_layout(height=800, width=800, title='Graphiques des séries temporelles brutes')
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
    
def lissage_moyenne_mobile(df, list_col, fenetre_des_moyennes, graphiques_Y_N):

    for col in list_col:
        last_observations = df[col].tail(fenetre_des_moyennes)
        last_observations_smoothed = pd.Series(last_observations[-3:].rolling(window=10).mean())
        series_smoothed =  df[col].copy()
        series_smoothed.iloc[-3:] = last_observations_smoothed.fillna(last_observations.mean())

        if graphiques_Y_N == 'Oui':
            fig = make_subplots(rows=1, cols=1)

            fig.add_trace(go.Scatter(x=df.index, y=df[col], name='Original series : '+col))
            fig.add_trace(go.Scatter(x=df.index, y=series_smoothed, name='Smoothed series'))

            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='Values')

            fig.update_layout(title='Smoothed series with last 3 observations replaced by their moving average', height=500, width=700)

            fig.show()

        df[col] = series_smoothed 

def logit(p):
    return np.log(p) - np.log(1 - p)

def import_and_series_transformation(drop_na, list_col_saison, list_col_lissage, logit_choix) :

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
    macro["td"]=td


    # drop les valeurs manquantes (selon la colonne du TD)
    if drop_na == 'Oui':
        macro = macro.dropna(subset=['td'], how='all', inplace=True)

    # retirons la saisonnalités des variables choisies
    macro = remove_season(macro, list_col_saison)

    # Variation mensuelle
    macro[['RGDP_tri', 'HICP_tri', 'IRLT_tri',  'RREP_tri', 'UNR_tri']] = (macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']] - macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']].shift(1)) / macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']].shift(1)

    # Variation annuelle
    macro[['RGDP_ann', 'HICP_ann', 'IRLT_ann',  'RREP_ann', 'UNR_ann']] = (macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']] - macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']].shift(4)) / macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']].shift(4)

    # retirons les variables brutes
    col_brutes = ['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']
    macro=macro.drop(col_brutes, axis=1)

    # Lissage par moyenne mobile des variables indiquées dans la liste 'columns'
    window_size = 10
    lissage_moyenne_mobile(macro, list_col_lissage, window_size, 'N')

    # appliquer le logit sur le taux de défaut ou pas
    if logit_choix == 'Oui':
        macro['DD'] = macro['td'].apply(logit)
        macro['td'] = macro['DD']
        macro = macro.drop('DD', axis=1)
        
    return macro


def import_and_series_transformation_and_graph(drop_na, list_col_saison, list_col_lissage, logit_choix, variable_graph) :

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
    macro["td"]=td


    # drop les valeurs manquantes (selon la colonne du TD)
    if drop_na == 'Oui':
        macro.dropna(subset=['td'], how='all', inplace=True)

    # retirons la saisonnalités des variables choisies
    macro = remove_season(macro, list_col_saison)

    # Variation mensuelle
    macro[['RGDP_tri', 'HICP_tri', 'IRLT_tri',  'RREP_tri', 'UNR_tri']] = (macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']] - macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']].shift(1)) / macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']].shift(1)

    # Variation annuelle
    macro[['RGDP_ann', 'HICP_ann', 'IRLT_ann',  'RREP_ann', 'UNR_ann']] = (macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']] - macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']].shift(4)) / macro[['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']].shift(4)

    # retirons les variables brutes
    col_brutes = ['RGDP', 'HICP', 'IRLT',  'RREP', 'UNR']
    macro=macro.drop(col_brutes, axis=1)

    # Lissage par moyenne mobile des variables indiquées dans la liste 'columns'
    window_size = 10
    lissage_moyenne_mobile(macro, list_col_lissage, window_size, 'N')

    # appliquer le logit sur le taux de défaut ou pas
    if logit_choix == 'Oui':
        macro['DD'] = macro['td'].apply(logit)
        macro['td'] = macro['DD']
        macro = macro.drop('DD', axis=1)
        
    font_size = 16
    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create two traces using go.Scatter
    trace1 = go.Scatter(x=macro.index, y=macro['td'], name='DR', line=dict(color='darkred'))
    trace2 = go.Scatter(x=macro.index, y=macro[variable_graph], name=variable_graph, line=dict(color='blue'))

    # Add traces to subplot grid, setting secondary_y=True for the second trace
    subfig.add_trace(trace1, secondary_y=False)
    subfig.add_trace(trace2, secondary_y=True)

    # Customize axis labels and titles
    subfig.update_xaxes(title_text='Date')
    subfig.update_yaxes(title_text='DR', secondary_y=False)
    subfig.update_yaxes(title_text=variable_graph, secondary_y=True)

    # Customize layout
    subfig.update_layout(title=variable_graph + " : Série macro transformée, avec le taux de défaut", title_font=dict(size=font_size+4), legend=dict(x=1, y=1, font=dict(size=font_size)))

    # Display plot using st.plotly_chart
    st.plotly_chart(subfig)

    return macro

def drop_trimestrielle_or_annuelle_if_wanted(df, choix) :
    """
    What kind of variables do you wish to keep :
    choix = {'both', 'Trimestrielles', 'Annuelles'}"""

    if choix == 'Trimestrielles':
        df = df.filter(regex='^(td|.*_tri)$')
    elif choix == 'Annuelles':
        df = df.filter(regex='^(td|.*_ann)$')
    else :
        pass
    return df

def list_str(list_to_change):
    columns = [col.strip("[]'") for col in list_to_change]
    return columns
######################################################################################

####################   Stationnarité des variables macro   ###########################

######################################################################################

def ljung_box(df, drop_td):
    if drop_td == 'Yes':
        macro_Box = df.drop('td', axis=1).dropna()
    else :
        macro_Box = df.dropna()
    results = pd.DataFrame(columns=['Variable', 'Ljung-Box Statistic', 'p-value','Résultat'])
    for col in macro_Box:
        res = ARIMA(macro_Box[col], order=(1,0,1)).fit()
        # print(sm.stats.acorr_ljungbox(res.resid, lags=[5]))
        results.loc[len(results)] = [col, acorr_ljungbox(res.resid, lags=[5]).iloc[0, 0], acorr_ljungbox(res.resid, lags=[5]).iloc[0, 1], acorr_ljungbox(res.resid, lags=[5]).iloc[0, 1] < 0.05]
    results['Résultat'] = results['Résultat'].replace(False, 'Les résidus sont indépendants')
    results['Résultat'] = results['Résultat'].replace(True, "Les résidus NE sont PAS indépendants : il y a de l'autocorrélation dans la série")
    
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


def kpss_test(df, regression, alpha='5%'):
    # Initialisation du DataFrame de résultats
    results = pd.DataFrame(columns=['Test', 'Variable', 'Regression', 'Test Statistic', 'p-value', 'Lags Used', 'Résultat'])

    # Boucle sur chaque colonne de la DataFrame
    for col in df.columns:
        # Calcul du test KPSS pour la série temporelle
        kpss_result = kpss(df[col], regression=regression)
        
        # Ajout des résultats du test ADF dans le DataFrame de résultats
        results.loc[len(results)] = ['KPSS', col, regression, round(kpss_result[0], 3), round(kpss_result[1], 3), kpss_result[2], kpss_result[0] < kpss_result[3][alpha]]
    
    # Remplacement des valeurs booléennes par des chaînes de caractères lisibles
    results['Résultat'] = results['Résultat'].replace(False, 'Non Stationnaire')
    results['Résultat'] = results['Résultat'].replace(True, 'Stationnaire')

    return results

def PhillipsPerron_test(df,regression, alpha='5%'):
    # Initialisation du DataFrame de résultats
    results = pd.DataFrame(columns=['Test' ,'Variable', 'Regression', 'Test Statistic', 'p-value', 'Lags Used', 'Résultat'])

    # Boucle sur chaque colonne de la DataFrame
    for col in df.columns:
        # Calcul du test ADF pour la série temporelle
        pp_result = PhillipsPerron(df[col], trend=regression)
        # Ajout des résultats du test ADF dans le DataFrame de résultats
        results.loc[col] = ['PP', col, regression, round(pp_result.stat, 3), round(pp_result.pvalue, 3) , pp_result.lags, pp_result.pvalue < 0.05]
    
    # Remplacement des valeurs booléennes par des chaînes de caractères lisibles
    results['Résultat'] = results['Résultat'].replace(False, 'Non Stationnaire')
    results['Résultat'] = results['Résultat'].replace(True, 'Stationnaire')

    return results

def AugmenteDickeyFuller_test(df,regression, alpha='5%'):
    # Initialisation du DataFrame de résultats
    results = pd.DataFrame(columns=['Test', 'Variable', 'Regression', 'Test Statistic', 'p-value', 'Lags Used', 'Résultat'])

    # Boucle sur chaque colonne de la DataFrame
    for col in df.columns:
        # Calcul du test ADF pour la série temporelle
        adf_result = adfuller(df[col], regression=regression)
        # Ajout des résultats du test ADF dans le DataFrame de résultats
        results.loc[col] = ['ADF', col, regression, round(adf_result[0], 3), round(adf_result[1], 3) , adf_result[2], adf_result[1] < 0.05]
    
    # Remplacement des valeurs booléennes par des chaînes de caractères lisibles
    results['Résultat'] = results['Résultat'].replace(False, 'Non Stationnaire')
    results['Résultat'] = results['Résultat'].replace(True, 'Stationnaire')

    return results


def highlight_variables(row):
    if row['Variable'] == 'HICP_ann':
        return ['background-color: linen']*len(row)
    if row['Variable'] == 'HICP_tri':
        return ['background-color: peru']*len(row)
    if row['Variable'] == 'IRLT_ann':
        return ['background-color: seashell']*len(row)
    if row['Variable'] == 'IRLT_tri':
        return ['background-color: palegreen']*len(row)
    if row['Variable'] == 'RGDP_ann':
        return ['background-color: azure']*len(row)
    if row['Variable'] == 'RGDP_tri':
        return ['background-color: lightblue']*len(row)
    if row['Variable'] == 'RREP_ann':
        return ['background-color: lavender']*len(row)
    if row['Variable'] == 'RREP_tri':
        return ['background-color: thistle']*len(row)
    if row['Variable'] == 'UNR_ann':
        return ['background-color: lavenderbush']*len(row)
    if row['Variable'] == 'UNR_tri':
        return ['background-color: lightpink']*len(row)
    else:
        return ['background-color: chocolate']*len(row)

    

def highlight_value(val):
    if val == 'Les résidus sont indépendants':
        return 'color: white; background-color: skyblue'
    else:
        return 'color: white; background-color: mediumvioletred'


def highlight_result(val):
    if val == 'Stationnaire':
        return 'color: white; background-color: skyblue'
    else:
        return 'color: white; background-color: mediumvioletred'


def concat_tests_and_sort (kpss, pp, adf):
    results_of_tests = pd.concat([kpss, pp, adf], ignore_index=True)
    results_of_tests = results_of_tests.drop('Test Statistic', axis=1)
    results_of_tests = results_of_tests.sort_values(['Variable', 'Regression'], ascending=[True, False])
    return results_of_tests

def draw_conclusion_stationary(results_all_tests, nb_variables_studies):

    # Group the dataframe by 'Variable' and 'Résultat', then get the size of each group
    result_counts = results_all_tests.groupby(['Variable', 'Résultat']).size().reset_index(name='Count')

    # Filter the groups where 'Résultat' is 'Stationary'
    result_counts = result_counts[result_counts['Résultat'] == 'Stationnaire']

    # Drop the 'Résultat' column from the result_counts dataframe
    result_counts = result_counts.drop('Résultat', axis=1)


    # Rename the 'Count' column to 'Stationary_count'
    result_counts = result_counts.rename(columns={'Count': 'Stationary_count'})

    # Merge the result_counts dataframe with the original results_all_tests dataframe
    results_all_tests = results_all_tests.merge(result_counts, on='Variable', how='left')

    # Group the dataframe by 'Variable', then get the count of rows for each group
    row_counts = results_all_tests.groupby('Variable').size()

    # Divide the 'Stationary_count' column by the row counts for each variable
    results_all_tests['Stationary_ratio'] = results_all_tests['Stationary_count'] / (len(results_all_tests)/nb_variables_studies)

    # Fill NaN values with 0
    results_all_tests['Stationary_ratio'] = results_all_tests['Stationary_ratio'].fillna(0)

    results_all_tests['Proportion de conclusion positive pour la stationnarité (%)'] = (round(results_all_tests['Stationary_ratio'], 4) * 100)

    # decision finale sur la stationnarité de la variable
    results_all_tests.loc[results_all_tests['Proportion de conclusion positive pour la stationnarité (%)']>50, 'Décision Finale'] = 'Stationnaire'
    results_all_tests.loc[results_all_tests['Proportion de conclusion positive pour la stationnarité (%)']<=50, 'Décision Finale'] = 'Non Stationnaire'

    return results_all_tests

def readable_conclusions(results_all_tests):
    
    # Drop non useful columns
    col_drop = ['Test','p-value', 'Regression','Regression', 'Lags Used', 'Résultat', 'Stationary_count','Stationary_ratio']
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
    """On doit faire le test de Newey-West, en utilisant la série DRt - Beta*DRt-1. On passe par les tests KPSS, PP et ADF """
    
    #creation du dataframe qu'on utilisera à chaque fois
    df = pd.DataFrame()
    df['td'] = dataframe['td']
    df['td_minus1'] = df['td'].shift(1)
    df = df.dropna()
    
    # create an empty DataFrame for the results with columns for betax, stat, crit, and regression type
    df_results = pd.DataFrame(columns=['betax', 'stat', 'crit', 'regression type']) 
    fig = go.Figure()

    betax = []
    pvalues = []
    crit = []
    stat = []

    for beta in range(100):
        df['DRt_beta'] = df['td'] - (beta/100)*df['td_minus1'] #creation de la série qu'on va utiliser pour mener le test
        
        if test == 'KPSS':
            reg = 'c'
            kpss_result = kpss(df["DRt_beta"], regression=reg)
            stat.append(kpss_result[0])
            pvalues.append(kpss_result[1])
            crit.append(kpss_result[3]['5%']) # récupérer la valeur critique
        elif test == 'PP':
            reg = 'c'
            pp_result = PhillipsPerron(df['DRt_beta'], trend=reg)
            stat.append(pp_result.stat)
            pvalues.append(pp_result.pvalue)
            crit.append(pp_result.critical_values['5%']) # récupérer la valeur critique
        else:
            reg = 'c'
            adf_result = adfuller(df['DRt_beta'], regression=reg)
            stat.append(adf_result[0])
            pvalues.append(adf_result[1])
            crit.append(adf_result[4]['5%']) # récupérer la valeur critique
        df=df.drop('DRt_beta', axis=1)
        betax.append(beta/100)
    
    # create a new DataFrame for the current regression type --> utile dans le cas où je mets plusieurs types de regression à tester
    df_results = pd.DataFrame({'betax': betax, 'stat': stat, 'crit': crit, 'regression type': [reg] * 100})

    # add the results to the main DataFrame

    #fig.add_trace(go.Scatter(x=betax, y=stat, name=test + " Statistics", line=dict(color='mediumvioletred')))
    #fig.add_trace(go.Scatter(x=betax, y=crit, name="valeurs critiques", line=dict(color='crimson')))
    fig.add_trace(go.Scatter(x=betax, y=[0.05] * np.shape(crit)[0], name="Seuil de 5%", line=dict(color='darkslategrey')))
    fig.add_trace(go.Scatter(x=betax, y=pvalues, name="pvalue", line=dict(color='skyblue')))
    fig.update_xaxes(title_text="Beta values")
    fig.update_yaxes(title_text="Test statistique Newey & West sur le Beta")
    fig.update_layout(title="Valeurs de Beta selon le test " + test, title_font=dict(size=16))
    st.plotly_chart(fig)
    
    return df_results

def AAAAnewey_west_tests_find_beta(dataframe, test):
    """On doit faire le test de Newey-West, en utilisant la série DRt - Beta*DRt-1. On passe par les tests KPSS, PP et ADF """
    
    #creation du dataframe qu'on utilisera à chaque fois
    df = pd.DataFrame()
    df['td'] = dataframe['td']
    df['td_minus1']=df['td'].shift(1)
    df = df.dropna()
    
    # create an empty DataFrame for the results with columns for betax, stat, crit, and regression type
    df_results = pd.DataFrame(columns=['betax', 'stat', 'crit', 'regression type']) 
    fig, ax = plt.subplots()
    betax = []
    pvalues = []
    crit = []
    stat = []

    for beta in range(100):
        df['DRt_beta'] = df['td'] - (beta/100)*df['td_minus1'] #creation de la série qu'on va utiliser pour mener le test
        
        if test == 'KPSS':
            reg = 'c'
            kpss_result = kpss(df["DRt_beta"], regression=reg)
            stat.append(kpss_result[0])
            pvalues.append(kpss_result[1])
            crit.append(kpss_result[3]['5%']) # récupérer la valeur critique
        elif test == 'PP':
            reg = 'c'
            pp_result = PhillipsPerron(df['DRt_beta'], trend=reg)
            stat.append(pp_result.stat)
            pvalues.append(pp_result.pvalue)
            crit.append(pp_result.critical_values['5%']) # récupérer la valeur critique
        else:
            reg = 'c'
            adf_result = adfuller(df['DRt_beta'], regression=reg)
            stat.append(adf_result[0])
            pvalues.append(adf_result[1])
            crit.append(adf_result[4]['5%']) # récupérer la valeur critique
        df=df.drop('DRt_beta', axis=1)
        betax.append(beta/100)
    
    # create a new DataFrame for the current regression type --> utile dans le cas où je mets plusieurs types de regression à tester
    df_results = pd.DataFrame({'betax': betax, 'stat': stat, 'crit': crit, 'regression type': [reg] * 100})
    # add the results to the main DataFrame
    #df_results = pd.concat([df_results, results_df_temporary], ignore_index=True)

    ax.plot(betax,stat,color='mediumvioletred' ,label=test + " Statistics")
    ax.plot(betax,crit, color='crimson', label="valeurs critiques")
    ax.plot(betax,[0.05] * np.shape(crit)[0], color='darkslategrey',label="Seuil de 5%")
    ax.plot(betax,pvalues,color='skyblue', label="pvalue")
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
    for col in macro_corr.filter(regex='_ann').columns:
        # calculer les Lags 
        for lag_number in range(1, nb_lag_var_ann+1):
            macro_corr[col+"_lag"+str(lag_number)] = macro_corr[col].shift(lag_number)

    for col in macro_corr.filter(regex='_tri').columns:
        # calculer les Lags 
        for lag_number in range(1, nb_lag_var_tri+1):
            macro_corr[col+"_lag"+str(lag_number)] = macro_corr[col].shift(lag_number)
    for col in macro_corr.filter(regex='td').columns:
        for lag_number in range(1, nb_lag_var_td+1):
                macro_corr[col+"_lag"+str(lag_number)] = macro_corr[col].shift(lag_number)
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
    
    """ on choisit le seuil de correlation qu'on veut, puis on crée une liste avec le nom des variables qu'on garde"""
    
    df_var_VERY_corr_to_DR = df[df['Kendall Correlations']>seuil]
    list_var_in_model = df_var_VERY_corr_to_DR.index.tolist()
    return list_var_in_model


def commpute_corr_matrix_var_macro(macro_lagged, list_var_over_threshold):
    macro_lagged = macro_lagged.dropna()
    corr_df_macro = pd.DataFrame(index=list_var_over_threshold, columns=list_var_over_threshold)
    for col1 in macro_lagged[list_var_over_threshold]:
        for col2 in macro_lagged[list_var_over_threshold]:
            corr, _ = kendalltau(macro_lagged[list_var_over_threshold][col1], macro_lagged[list_var_over_threshold][col2])
            corr_df_macro.loc[col1, col2] = corr
    return corr_df_macro

def find_high_corr_pairs(corr_matrix, threshold):
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    return high_corr_pairs

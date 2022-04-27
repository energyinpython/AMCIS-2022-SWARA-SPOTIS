import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm

from weighting_methods import merec, swara_weighting
from rank_preferences import rank_preferences
from create_dictionary import Create_dictionary
from correlations import weighted_spearman
from spotis import SPOTIS


def main():
    path = 'dataset'
    m = 30
    n = 11

    str_years = [str(y) for y in range(2015, 2021)]
    list_alt_names = ['A' + str(i) for i in range(1, m + 1)]
    list_alt_names_latex = [r'$A_{' + str(i + 1) + '}$' for i in range(0, m)]
    preferences = pd.DataFrame(index = list_alt_names)
    rankings = pd.DataFrame(index = list_alt_names)
    presentation = pd.DataFrame(index = list_alt_names)

    averages = np.zeros((m, n))

    spotis = SPOTIS()
    
    method_name = 'SPOTIS'
    for el, year in enumerate(str_years):
        file = 'data_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Country')
        
        df_data = data.iloc[:len(data) - 1, :]
        # types
        types = data.iloc[len(data) - 1, :].to_numpy()
        
        list_of_cols = list(df_data.columns)
        # matrix
        matrix = df_data.to_numpy()
        averages += matrix
        # weights
        weights = merec(matrix, types)

        # SPOTIS preferences are sorted in ascending order
        bounds_min = np.amin(matrix, axis = 0)
        bounds_max = np.amax(matrix, axis = 0)
        bounds = np.vstack((bounds_min, bounds_max))
        pref = spotis(matrix, weights, types, bounds)
        rank = rank_preferences(pref, reverse = False)
        
        preferences[year] = pref
        rankings[year] = rank

    country_names = list(data.index)
    country_names[10] = country_names[10][:7]
    
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Set2(i))
    for i in range(10):
        color.append(cm.tab10(i))
    for i in range(8):
        color.append(cm.Pastel2(i))
    
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": "-"})
    ticks = np.arange(1, m + 2, 2)
    
    x1 = np.arange(0, len(str_years))
    plt.figure(figsize = (7, 6))
    for i in range(rankings.shape[0]):
        c = color[i]
        plt.plot(x1, rankings.iloc[i, :], color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(list_alt_names_latex[i] + ' ' + country_names[i], (x_max - 0.2, rankings.iloc[i, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel("Year", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.xticks(x1, str_years, fontsize = 12)
    plt.yticks(ticks, fontsize = 12)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title(method_name + ' rankings')
    plt.tight_layout()
    plt.savefig('results/rankings_years.png')
    plt.show()

    
    # AVERAGES for average data
    matrix_average = averages / len(str_years)
    # average weights
    weights_average = merec(matrix_average, types)

    # SPOTIS AVERAGE
    bounds_min = np.amin(matrix_average, axis = 0)
    bounds_max = np.amax(matrix_average, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))
    pref_average = spotis(matrix_average, weights_average, types, bounds)
    rank_average = rank_preferences(pref_average, reverse = False)

    # SWARA weighting for determining periods' significance
    s = np.ones(len(str_years) - 1) * 0.5
    new_s = np.insert(s, 0, 0)
    swara_weights = swara_weighting(new_s)[::-1]

    # save SWARA weights to csv
    df_weights = pd.DataFrame(swara_weights.reshape(1, -1), index = ['Weights'], columns = str_years)
    df_weights.to_csv('results/weights_swara.csv')

    matrix_swara = preferences.to_numpy()
    
    
    # SPOTIS SWARA
    swara_types = np.ones(len(str_years)) * (-1)
    bounds_min = np.amin(matrix_swara, axis = 0)
    bounds_max = np.amax(matrix_swara, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))
    pref_swara = spotis(matrix_swara, swara_weights, swara_types, bounds)
    rank_swara = rank_preferences(pref_swara, reverse = False)

    # save results
    
    preferences['AVERAGE'] = pref_average
    rankings['AVERAGE'] = rank_average

    preferences['SWARA'] = pref_swara
    rankings['SWARA'] = rank_swara

    presentation['AVG'] = rank_average
    presentation['TSS'] = rank_swara
    presentation.to_csv('results/presentation.csv')
    
    preferences = preferences.rename_axis('Ai')
    preferences.to_csv('results/scores.csv')

    rankings = rankings.rename_axis('Ai')
    rankings.to_csv('results/rankings.csv')
    
    # correlations
    method_types = list(rankings.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            dict_new_heatmap_rw[j].append(weighted_spearman(rankings[i], rankings[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_rw.to_csv('results/df_new_heatmap_rw.csv')


if __name__ == '__main__':
    main()
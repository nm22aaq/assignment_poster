# importing important libraries
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import sklearn.metrics as skmet
import sklearn.preprocessing as prep
import scipy.optimize as opt
import itertools



def create_df(df1, df_gdp,yr):
    '''
    this function pre-processes the data and creating a new dataframe

    parameters:
        df1 - dataframe in worldbank data formate
        df_gdp - dataframe of the gdp per capita dataset
        yr - year for which we want to extract data
    
    returns:
        final dataframe
    '''
    gdp_df = df_gdp[yr].copy()
    gdp_df['DATA'] = df1[yr].to_numpy()
    gdp_df.rename(columns = {yr[0]:'GDP'}, inplace = True)
    gdp_df.dropna(inplace=True)

    #putting the GDP on the y-axis
    first_column = gdp_df.pop('DATA')
    gdp_df.insert(0, 'DATA', first_column)
    
    return gdp_df




def read_twb_data(filename, indicator_code):
    ''' 
    This function reads the data(worldbank format) from the given csv file in pandas Data Frame format.
    Because the first 4 rows in the world bank data format files doesn't include useful information so we skip them.
    Also extra columns are dropped.
    Finally the the dataframes with Years as column and the data frame with Country Names are both returned

    Parameters:
        filename - path of the file names which contains the dataset in worldbank data format
        indicator_code - the code for climate change indicator
    Returns:
        df_years - the data frame with Years as columns
        df_countries - the data frame with Countries as Columns for that specific indicator

    '''

    # It reads the data in a data frame from a .csv file, skips the first 4 rows and sets the first column as index
    df_wbdata = pd.read_csv(filename, skiprows=4)
    
    # It only extracts data for a specific indicator
    wbdata = df_wbdata.loc[df_wbdata["Indicator Code"] == indicator_code, :]

    wbdata.index = wbdata.iloc[:, 0]

    # Removes the Extra Columns which are not useful for us
    df_years = wbdata.drop(["Country Name", "Indicator Name", "Country Code", "Indicator Code" , "Unnamed: 66"], axis=1)

    # Transpose the DataFrame so that it has countries as its columns
    df_countries = df_years.T.copy()

    # Return both the Data frames
    return df_years, df_countries


def calculate_and_plot_Kmeans(ax, data, ncluster, xlabel='', ylabel=''):

    '''
    this function trains kmeans on the dataset with ncluster clusters and then plots it

    parameters
        ax - axis of the plot to draw on 
        data - a dataset 
        ncluster - the number of clusters for which the kmeans will be trained
        xlabel - the label on x-axis of the plot
        ylabel - the label on the y-axis of plot

    returns:
        silhoutte - the silhoutte score of the clustering
    '''

    # creating the K-Means model
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=10);

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(data) # fit done on x,y pairs
    labels = kmeans.labels_ # labels is the number of the associated clusters of (x,y)→points

    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    
    # calculate the silhoutte score
    silhoutte = skmet.silhouette_score(data, labels)

    # plot using the labels to select colour
    #plt.figure(figsize=(10.0, 10.0))
    col_names=data.columns
    
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for l in range(ncluster): # loop over the different labels
        ax.plot(data[labels==l][col_names[0]], data[labels==l][col_names[1]], "o", markersize=3, color=col[l])
    #

    # show cluster centres
    for ic in range(ncluster):
        xc, yc = cen[ic,:]
        ax.plot(xc, yc, "dk", markersize=10)
        ax.text(0.5, 0.95, 'Silhoutte Score: ' + str(round(silhoutte,4)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    # plt.xlabel(col_names[0] if xlabel=='' else xlabel)
    # plt.ylabel(col_names[1] if ylabel=='' else ylabel)
    # plt.show()

    return silhoutte


def show_all_clusters(data, gdp_data, meta, title=""):
    '''
    this function plots a figure which consists of several plots
    '''
    figure, axis = plt.subplots(2, 3, sharex=True, sharey=True)
    figure.set_size_inches(12,5)

    z= itertools.product(range(2), range(3))
    sscore=[]
    for i,axi in zip(range(1970,2021,10), z):
        yr = [str(i)]
        df = create_df(data, gdp_data, yr)
        
        if df.shape[0] != 0:
            sscore.append(calculate_and_plot_Kmeans(axis[axi[0],axi[1]], df,3))
        else:
            sscore.append(0)
        axis[axi[0], axi[1]].set_title(yr[0])

    max_index = np.argmax(sscore)
    ax = axis[max_index//3,(max_index%3)]
    ax.text(0.5, 0.95, 'Silhoutte Score: ' + str(round(sscore[max_index],4)), bbox=dict(facecolor='red', alpha=0.3), transform=ax.transAxes, verticalalignment='center', horizontalalignment='center', color='red')
       
    figure.suptitle(title)
    figure.supxlabel(meta['Title'])
    figure.supylabel("GDP")
    for ax in figure.get_axes():
        ax.label_outer()
    plt.show();




def create_df_country_gdp_data(df_gdp, country):
    df = df_gdp.loc[df_gdp['Country Name'] == country, df_gdp.columns[4:-1]]
    df = df.transpose()
    df.columns = ["GDP"]
    df = df.reset_index()
    df = df.rename(columns={"index": "Year"})
    df.dropna(inplace=True)
    return df

def create_df_country_worldbank_data(data, country, col_name):
    df = data.loc[data.index == country, :]
    df = df.transpose()
    df.columns = [col_name]
    df = df.reset_index()
    df = df.rename(columns={"index": "Year"})
    df.dropna(inplace=True)
    return df

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


def draw_logistic_fit_axes(ax, df, country, col_type):
    # convert the Year column to numeric values 
    df["Year"] = pd.to_numeric(df["Year"])

    # fitting the logistic 
    param, covar = opt.curve_fit(logistic, df["Year"], df[col_type], p0=(max(df[col_type]), 0.03, 2000.0),maxfev=20000)
    #param, covar = opt.curve_fit(logistic, df["Year"], df[col_type], p0=None)
    sigma = np.sqrt(np.diag(covar))

    #print("parameters:", param)
    #print("std. dev.", sigma)
    df["fit"] = logistic(df["Year"], *param)
    ax.plot(df["Year"],df[[col_type,'fit']])
    ax.set_title(country)
    #plt.show()


def show_all_logistic_fits_gdp(data,countries):
    figure, axis = plt.subplots(2, 3, sharex=True, sharey=True)
    figure.set_size_inches(14,4)

    z= itertools.product(range(2), range(3))
    for country,axi in zip(countries, z):
        df = create_df_country_gdp_data(data, country)
        if df.shape[0] == 0:
            continue
        df.dropna(inplace=True)
        draw_logistic_fit_axes(axis[axi[0],axi[1]], df, country, "GDP")
        
    figure.suptitle("Logistic Curve Fit for GDP Per Capita of Different Countries")
    figure.supxlabel("Year")
    figure.supylabel("GDP")
    for ax in figure.get_axes():
        ax.label_outer()
    plt.show();

def show_all_logistic_fits_worldbank_data(data,countries, indicator, title):
    figure, axis = plt.subplots(2, 3, sharex=True, sharey=True)
    figure.set_size_inches(14,4)

    z= itertools.product(range(2), range(3))
    for country,axi in zip(countries, z):
        df = create_df_country_worldbank_data(data, country, indicator)
        if df.shape[0] == 0:
            continue
        df.dropna(inplace=True)
        draw_logistic_fit_axes(axis[axi[0],axi[1]], df, country, indicator)
        
    figure.suptitle(title)
    figure.supxlabel("Year")
    figure.supylabel(indicator)
    for ax in figure.get_axes():
        ax.label_outer()
    plt.show();




if __name__ == "__main__":


    df_gdp = pd.read_csv("data/GDPPC.csv", skiprows=4)



    #The path of the file for world bank Data
    filename = "data/wbdata.csv"

    # The codes and titles for different Indicators of climate change in world bank data
    co2_emission= {"Code": "EN.ATM.CO2E.PC", "Title": "CO2 emissions (metric tons per capita)"} # CO2 emissions (metric tons per capita)
    electriciy_production= {"Code": "EG.ELC.PETR.ZS", "Title": "Electricity production from oil sources (% of total)"} # Electricity production from oil sources (% of total)
    forest_area = {"Code": "AG.LND.FRST.ZS", "Title": "Forest area (% of land area)"} # Forest area (% of land area)
    pop_growth = {"Code": "SP.POP.GROW", "Title": "Population growth (annual %)"}     # Population growth (annual %)
    urban_pop_growth = {"Code": "SP.URB.GROW", "Title": "Urban population growth (annual %)"} #Urban population growth (annual %)
    agri_land = {"Code": "AG.LND.AGRI.ZS", "Title": "Agricultural land (% of land area)"} #Agricultural land (% of land area)

    # Read the data from the world bank data file
    df_co2_emission, df_co2_emission_c  = read_twb_data(filename, co2_emission["Code"])
    df_electricity_production, df_electricity_production_c  = read_twb_data(filename, electriciy_production["Code"])
    df_forest_area, df_forest_area_c = read_twb_data(filename, forest_area["Code"])
    df_pop_growth, df_pop_growth_c = read_twb_data(filename, pop_growth["Code"])
    df_urban_pop_growth, df_urban_pop_growth_c = read_twb_data(filename, urban_pop_growth["Code"])
    df_agri_land, df_agri_land_c = read_twb_data(filename, agri_land["Code"])



    # plotting the curve fits

    # the list of countries for which we want to plot the logistic fit curve
    countries = ['Australia','United Kingdom', "United States", 'France', 'Brazil', 'China']
    # plot fits of the gdp data
    show_all_logistic_fits_gdp(df_gdp, countries)
    #plot the figure with fits of the co2 emission
    show_all_logistic_fits_worldbank_data(df_co2_emission, countries, "CO2_EMISSION", title="Logistic Curve Fit of "+ co2_emission['Title'])
    #plot the logistic curve fits of the population growth
    show_all_logistic_fits_worldbank_data(df_pop_growth, countries, "POP_GROWTH", title="Logistic Curve Fit of " + pop_growth['Title'])
    
    show_all_logistic_fits_worldbank_data(df_agri_land, countries, "AGRI_LAND", title="Logistic Curve Fit of " + agri_land['Title'])


    
    # here we show all the figures with kmeans clusters plotted on them
    show_all_clusters(df_pop_growth, df_gdp,  pop_growth, "Clustering of GDP per Capita and " + pop_growth['Title'])
    show_all_clusters(df_agri_land, df_gdp,agri_land, "Clustering of GDP per Capita and " + agri_land['Title'])
    show_all_clusters(df_forest_area, df_gdp,forest_area, "Clustering of GDP per Capita and " + forest_area['Title'])
    show_all_clusters(df_co2_emission, df_gdp,co2_emission, "Clustering of GDP per Capita and " + co2_emission['Title'])
    show_all_clusters(df_electricity_production, df_gdp, electriciy_production, "Clustering of GDP per Capita and " + electriciy_production['Title'])


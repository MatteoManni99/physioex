import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import math

concept_palette_protoae_shhs = [
    "#C0C0C0",  #// 0: # light grey
    "#CCCC00",  #// 1: dark yellow
    "#009900",  #// 2: dark green
    "#606060",  #// 3: grey
    "#1E90FF",  #// 4: sky blue
    "#A2C523",  #// 5: olive green
    "#909090",  #// 6: grey
    "#FF4500",  #// 7: orange red
    "#00FF00",  #// 8: bright green
    "#FF6347",  #// 9: tomato
    "#0000FF",  #// 10: blue
    "#FFA500",  #// 11: orange
    "#66FF66",  #// 12: light green
    "#DDA0DD",  #// 13: plum
    "#8A2BE2"   #// 14: blue violet
]

concept_palette_protoae_alldata = [
    "#0000FF",  #// 0: blue
    "#FFA500",  #// 1: orange
    "#009900",  #// 2: dark green
    "#D3D3D3",  #// 3: # light grey
    "#1E90FF",  #// 4: sky blue
    "#66FF66",  #// 5: light green
    "#606060",  #// 6: grey
    "#8A2BE2",  #// 7: blue violet
    "#A2C523",  #// 8: olive green
    "#DDA0DD",  #// 9: plum
    "#C0C0C0",  #// 10: # light grey
    "#FF4500",  #// 11: orange red
    "#CCCC00",  #// 12: dark yellow
    "#909090",  #// 13: grey
    "#CC8400"   #// 14: dark orange
]

class_palette = sns.color_palette([
    "#808080", # grey
    "#FF0000", # red
    "#FFA500", # orange
    "#00FF00", # green
    "#0000FF", # blue
])
sleep_stage_palette = ["#808080","#FF0000","#FFA500","#00FF00", "#0000FF"]
colormap = LinearSegmentedColormap.from_list('colormap_name', sleep_stage_palette, N=5)

def plot_pca_2d(data, colors, percentage=1, additional_points=None, num_additional_points=False, p_palette=None, title='PCA 2D', legend_title='Sleep Stages'):
    """
    Visualizes the data in a 2D PCA plot with options for coloring and additional points.

    This function performs Principal Component Analysis (PCA) to reduce the dimensionality of the input data 
    to two components for visualization. It allows for coloring the data points based on a given categorical 
    variable and includes optional additional points.

    Parameters:
    ----------
    data : numpy.ndarray or pandas.DataFrame
        A 2D array or DataFrame containing the data points to be visualized. Each row represents 
        an observation, and each column represents a feature.

    colors : array-like
        An array or list containing the colors for the points in the plot. This could represent 
        different categories or labels for the data points.

    percentage : float, optional
        A value between 0 and 1 indicating the fraction of the data to be sampled for plotting. 
        Defaults to 1, which means all data points are used.

    additional_points : numpy.ndarray or pandas.DataFrame, optional
        A 2D array or DataFrame containing additional points to be plotted on the PCA graph. 
        Each row represents a point. Defaults to None.

    num_additional_points : bool, optional
        If set to True, each additional point will be labeled with its index. Defaults to False.

    p_palette : list, optional
        A list of colors to be used for the points in the plot. If None, a default color palette 
        will be used. Defaults to None.

    title : str, optional
        The title of the plot. Defaults to 'PCA 2D'.

    legend_title : str, optional
        The title of the legend. Defaults to 'Sleep Stages'.

    Returns:
    -------
    None
        This function does not return any value. It generates and displays a plot directly.

    Example:
    --------
    import numpy as np
    
    # Create a sample dataset
    data = np.random.rand(100, 5)  # 100 samples, 5 features
    colors = np.random.choice(['Stage 1', 'Stage 2', 'Stage 3'], size=100)  # Random sleep stages
    
    # Define additional points (prototypes)
    additional_points = np.array([[0.5, 0.5, 0.5, 0.5, 0.5],
                                   [0.2, 0.2, 0.2, 0.2, 0.2]])
    
    # Call the PCA plotting function
    plot_pca_2d(data, colors, percentage=0.8, additional_points=additional_points, num_additional_points=True)

    Notes:
    ------
    - The function requires the `scikit-learn`, `numpy`, `pandas`, `matplotlib`, and `seaborn` 
      libraries for execution. Ensure that these libraries are installed in your environment.
    - The scatter plot shows the data points reduced to two dimensions, with explained variance 
      displayed on the axes.

    """
    # Perform PCA to reduce to 2 components
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    # Print the explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by component 1: {explained_variance[0]:.2f}")
    print(f"Explained variance by component 2: {explained_variance[1]:.2f}")
    
    # Sample a percentage of the data
    if (percentage<1):
        num_elements = int(len(data) * percentage)
        random_indices = np.random.choice(len(data), num_elements, replace=False)

        # Create new sampled arrays
        pca_result = pca_result[random_indices]
        colors = colors[random_indices]
    
    plt.figure(figsize=(10, 8))

    # Define a palette
    if p_palette is not None:
        palette = p_palette
    else:
        palette = sns.color_palette(["#808080","#FF0000","#FFA500","#00FF00", "#0000FF"])

    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=colors, palette=palette, alpha=0.6)
    
    # Check if additional points are provided
    if additional_points is not None:
        additional_points_scaled = scaler.transform(additional_points)
        additional_points_2d = pca.transform(additional_points_scaled)
        df_additional = pd.DataFrame(additional_points_2d, columns=['PC1', 'PC2'])
        df_additional['Cluster'] = 'Prototype'
        # Plot the additional points
        if num_additional_points is False:
            sns.scatterplot(data=df_additional, x='PC1', y='PC2', color='black', s=100, marker='X', edgecolor='white', legend='full',)
        # Plot the additional points with numbers
        else:
            sns.scatterplot(data=df_additional, x='PC1', y='PC2', color='black', s=10, marker='o', edgecolor='white', legend='full')
            for i, (x, y) in enumerate(zip(additional_points_2d[:, 0], additional_points_2d[:, 1])):
                plt.text(x, y, str(i), color='white', fontsize=12, ha='center', va='center', weight='bold',
                         path_effects=[path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
    
    plt.title(title)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f} variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f} variance)')
    plt.legend(title=legend_title)
    plt.grid(True)

    #plt.savefig('pca_2d_white.png', dpi=300)
    plt.show()

def plot_pca_3d(data, colors, title='PCA 3D', percentage=1):
    # Perform PCA to reduce to 3 components
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)
    
    # Print the explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by component 1: {explained_variance[0]:.2f}")
    print(f"Explained variance by component 2: {explained_variance[1]:.2f}")
    print(f"Explained variance by component 3: {explained_variance[2]:.2f}")
    
    # Sample a percentage of the data
    if (percentage<1):
        num_elements = int(len(data) * percentage)
        random_indices = np.random.choice(len(data), num_elements, replace=False)

        # Create new sampled arrays
        pca_result = pca_result[random_indices]
        colors = colors[random_indices]

    # Plotting the results in 3D with color
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=colors, cmap=colormap, alpha=0.8)
    
    ax.set_title(title)
    ax.set_xlabel(f'PC 1 ({explained_variance[0]:.2f} variance)')
    ax.set_ylabel(f'PC 2 ({explained_variance[1]:.2f} variance)')
    ax.set_zlabel(f'PC 3 ({explained_variance[2]:.2f} variance)')
    plt.show()

def plot_pca_3d_i(data, colors, percentage=1):
    # Perform PCA to reduce to 3 components
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)
    
    # Print the explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by component 1: {explained_variance[0]:.2f}")
    print(f"Explained variance by component 2: {explained_variance[1]:.2f}")
    print(f"Explained variance by component 3: {explained_variance[2]:.2f}")
    
    # Sample a percentage of the data
    if (percentage<1):
        num_elements = int(len(data) * percentage)
        random_indices = np.random.choice(len(data), num_elements, replace=False)

        # Create new sampled arrays
        pca_result = pca_result[random_indices]
        colors = colors[random_indices]

    # Create a DataFrame for Plotl
    df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'PC3': pca_result[:, 2],
        'Color': colors
    })
    # Plotting the results in 3D with color
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Color',
                        title='3D PCA of 32-dimensional data',
                        labels={'PC1': f'PC 1 ({explained_variance[0]:.2f} variance)',
                                'PC2': f'PC 2 ({explained_variance[1]:.2f} variance)',
                                'PC3': f'PC 3 ({explained_variance[2]:.2f} variance)'},
                        color_continuous_scale='Viridis')
    fig.update_layout(width=800,height=1000,)
    fig.update_traces(marker=dict(colorscale='Viridis'))
    fig.show()


def plot_parcord(data, colors, percentage=1, eps = 0.00001):
    if (percentage<1):
        num_elements = int(len(data) * percentage)
        random_indices = np.random.choice(len(data), num_elements, replace=False)

        # Create new sampled arrays
        data = data[random_indices]
        colors = colors[random_indices]

    df = pd.DataFrame(data[:,:])
    df['phase'] = colors
    df.iloc[0, 32] = 0
    df.iloc[1, 32] = 1
    df.iloc[2, 32] = 2
    df.iloc[3, 32] = 3
    df.iloc[4, 32] = 4
    # Creare il Parallel Coordinates Plot
    for i in range(32):
        if (df[i].max() - df[i].min()) < eps:
            print("Non used feature: ", i)
            
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df, 'phase', colormap=colormap)
    
    plt.title('Parallel Coordinates')
    plt.xlabel('Features')
    plt.ylabel('Value')
    plt.show()

def plot_silhouette_scores(data, n_clusters_range, percentage=1):
    silhouette_avgs = []
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    if (percentage<1):
        num_elements = int(len(data) * percentage)
        random_indices = np.random.choice(len(data), num_elements, replace=False)

        # Create new sampled arrays
        data = data[random_indices]

    for n_clusters in n_clusters_range:
        # Eseguire il clustering con K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calcolare l'indice di silhouette medio
        if len(set(cluster_labels)) > 1:  # Assicurati che ci siano almeno due cluster
            silhouette_avg = silhouette_score(data, cluster_labels)
        else:
            silhouette_avg = -1  # Se c'è solo un cluster, silhouette score non è definito
        
        silhouette_avgs.append(silhouette_avg)
        print(f"Silhouette n={n_clusters} - {round(silhouette_avg, 3)}\\\\")

    # Creare il grafico
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=n_clusters_range, y=silhouette_avgs, marker='o')
    plt.title('Average Silhouette Score for Different Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.grid(True)
    plt.show()
    
def plot_KMeans_with_PCA(data, n_clusters, additional_points=None):
    # Eseguire il clustering con K-means
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    
    # Ridurre la dimensionalità a 2D per la visualizzazione usando PCA
    pca_comp = 2
    pca = PCA(n_components=pca_comp)
    data_2d = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    for i in range(pca_comp):
        print(f"Explained variance by component {i+1}: {explained_variance[i]:.2f}")   
    
    # Creare un DataFrame per Seaborn
    df = pd.DataFrame(data_2d, columns=['PC1', 'PC2'])
    df['Cluster'] = cluster_labels

    # Plot dei cluster usando Seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='tab20', s=50, marker='o', legend='full')
    #protopyes
    if additional_points is not None:
        additional_points_scaled = scaler.transform(additional_points)
        additional_points_2d = pca.transform(additional_points_scaled)
        df_additional = pd.DataFrame(additional_points_2d, columns=['PC1', 'PC2'])
        df_additional['Cluster'] = 'Prototype'
        sns.scatterplot(data=df_additional, x='PC1', y='PC2', color='black', s=100, marker='X', edgecolor='white', legend='full',)

    plt.title(f'Cluster K-means (n_clusters = {n_clusters}) - PCA Projection')
    plt.xlabel(f'PC 1 ({explained_variance[0]:.2f} explained variance)')
    plt.ylabel(f'PC 2 ({explained_variance[1]:.2f} explained variance)')
    plt.legend(title='Cluster Label')
    plt.show()

def plot_clustering_accordind_to_additional_points(data, additional_points, silhouette = False):
    # Controllo se ci sono punti aggiuntivi
    if additional_points is None or len(additional_points) == 0:
        raise ValueError("I punti aggiuntivi devono essere forniti come centroidi.")

    # Standardizzare i dati
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    additional_points_scaled = scaler.transform(additional_points)
    
    # Assegnare ciascun punto al centroide più vicino
    distances = cdist(data, additional_points_scaled, metric='euclidean')
    cluster_labels = np.argmin(distances, axis=1)
    
    if silhouette:
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(f"Silhouette Coefficient: {silhouette_avg:.2f}")
        
    # Ridurre la dimensionalità a 2D per la visualizzazione usando PCA
    pca_comp = 2
    pca = PCA(n_components=pca_comp)
    data_2d = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    for i in range(pca_comp):
        print(f"Explained variance by component {i+1}: {explained_variance[i]:.2f}")   
    
    # Creare un DataFrame per Seaborn
    df = pd.DataFrame(data_2d, columns=['PC1', 'PC2'])
    df['Cluster'] = cluster_labels

    # Ridurre la dimensionalità dei punti aggiuntivi
    additional_points_2d = pca.transform(additional_points_scaled)
    
    # Plot dei cluster usando Seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='tab20', s=50, marker='o', legend='full')

    # Aggiungere i punti aggiuntivi come prototipi (centroidi)
    df_additional = pd.DataFrame(additional_points_2d, columns=['PC1', 'PC2'])
    df_additional['Cluster'] = 'Prototype'
    sns.scatterplot(data=df_additional, x='PC1', y='PC2', color='black', s=100, marker='X', edgecolor='white', legend='full')
    
    plt.title('Cluster using prototypes as centroid - PCA Projection')
    plt.xlabel(f'PC 1 ({explained_variance[0]:.2f} explained variance)')
    plt.ylabel(f'PC 2 ({explained_variance[1]:.2f} explained variance)')
    plt.legend(title='Cluster Label')
    plt.show()


def plot_distributions(concepts_target, y_max=None, file_path=None):
    """
    Plots the distribution of values for each column in the given array using histograms.

    This function visualizes the distribution of values for each column in a 2D numpy array, 
    providing statistical insights such as mean, standard deviation, maximum, and minimum for 
    each column. It generates histograms for each column and allows for saving the figure.

    Parameters:
    ----------
    concepts_target : numpy.ndarray
        A 2D numpy array where each column represents different concepts or prototypes, 
        and each row corresponds to observations or samples.

    y_max : float, optional
        The maximum limit for the y-axis of the histograms. If provided, it will standardize 
        the y-axis across all subplots. Defaults to None, allowing automatic scaling.

    file_path : str, optional
        The file path where the figure should be saved. If provided, the figure will be saved 
        in high resolution (300 dpi). Defaults to None, meaning the figure will not be saved.

    Returns:
    -------
    None
        This function does not return any value. It generates and displays a plot directly.

    Example:
    --------
    import numpy as np
    
    # Create a sample 2D array of concept activations
    concept_activations = np.random.rand(100, 10)  # 100 samples for 10 prototypes
    
    # Call the plotting function
    plot_distributions(concept_activations, y_max=50, file_path='distributions.png')

    Notes:
    ------
    - The function requires the `matplotlib` and `seaborn` libraries for plotting. Make sure 
      these libraries are installed in your environment.
    - The histograms show the density of values in each column, providing a visual understanding 
      of the distribution characteristics.

    """
    num_colonne = concepts_target.shape[1]

    # Calcola statistiche delle colonne
    media_colonne = np.mean(concepts_target, axis=0)
    std_colonne = np.std(concepts_target, axis=0)
    max_colonne = np.max(concepts_target, axis=0)
    min_colonne = np.min(concepts_target, axis=0)

    print("Media per ciascuna prototipo:", media_colonne)
    print("Deviazione standard per ciascuna colonna:", std_colonne)
    print("Max per ciascuna colonna:", max_colonne)
    print("Min per ciascuna colonna:", min_colonne)
    
    # Trova i limiti globali per l'asse x
    x_min = np.min(concepts_target)
    x_max = np.max(concepts_target)

    # Crea una figura con sottotitoli per ogni colonna
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))  # Adatta il numero di righe e colonne
    #fig.suptitle('Values Distribution')

    # Appiattisci l'array di assi per facilitare l'iterazione
    axs = axs.flatten()

    for i in range(num_colonne):
        sns.histplot(concepts_target[:, i], kde=False, ax=axs[i], stat='percent', color='blue', alpha=0.7, bins=75)  # Modifica il valore di binwidth a seconda delle tue esigenze
        axs[i].set_title(f'Prototype {i}')
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Density')
        
        # Imposta gli stessi limiti x per ogni subplot
        axs[i].set_xlim(x_min, x_max)

        if(y_max is not None):
            axs[i].set_ylim(0, y_max)

    # Regola il layout per evitare sovrapposizioni
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if file_path is not None:
        plt.savefig(file_path, dpi=300)
    plt.show()


def concept_boxplots(array, ylabel='Squared Error', title='Boxplots for Each Column of Squared Errors'):
    """
    Creates boxplots for each column of the input array and calculates relevant statistics.

    This function visualizes the distribution of data in each column of a given 2D numpy array 
    using boxplots. It also calculates and prints key statistics such as mean, standard deviation, 
    and outlier percentages.

    Parameters:
    ----------
    array : numpy.ndarray
        A 2D numpy array where each column represents a different concept activation or 
        squared error values. The function assumes the array has multiple rows for different 
        samples or observations.

    Returns:
    -------
    None
        This function does not return any value. It generates a boxplot and prints statistics 
        directly to the console.

    Example:
    --------
    import numpy as np
    
    # Create a sample 2D array of squared errors
    squared_errors = np.random.rand(100, 5)  # 100 samples with 5 concepts

    # Call the plotting function
    concept_boxplots(squared_errors)

    Notes:
    ------
    - The function requires the `seaborn` and `matplotlib` libraries for plotting. Make sure 
      these libraries are installed in your environment.
    - The boxplots help to visualize the central tendency and dispersion of data, while 
      outlier statistics provide insights into the presence of extreme values.

    """
    plt.figure(figsize=(8, 6))  # Adjust the size of the plot
    sns.boxplot(data=array)

    # Calculate the statistics for each column
    mean_per_column = np.mean(array, axis=0)
    std_per_column = np.std(array, axis=0)
    
    # Calculate Q1 (25th percentile), Q3 (75th percentile) and IQR for each column
    q1 = np.percentile(array, 25, axis=0)
    q3 = np.percentile(array, 75, axis=0)
    iqr = q3 - q1

    # Calculate the outlier thresholds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Count the number of outliers in each column
    outliers = ((array < lower_bound) | (array > upper_bound))
    outlier_percentage = np.sum(outliers, axis=0) / array.shape[0] * 100
    big_values = array > 0.03
    strong_outlier_percentage = np.sum(big_values, axis=0) / array.shape[0] * 100

    # Print the statistics
    print("Mean of each column:", mean_per_column)
    print("Standard deviation of each column:", std_per_column)
    print("Percentage of values outside the boxplot whiskers in each column:", outlier_percentage)
    print("Percentage of values greater then 0.03 in each column:", strong_outlier_percentage)

    # Customize plot labels and ticks
    plt.title(title)
    plt.xlabel('Concept Activations')
    plt.ylabel(ylabel)
    plt.yticks([0.03, 0.2,  0.4, 0.6, 0.8, 1])  # Adjust the step as necessary
    
    # Show the plot
    plt.show()

def plot_percentage_values(array):
    """
    Plots the percentage of points in an array that have at least N values greater than a threshold.

    This function computes and visualizes the percentage of rows in a given 2D numpy array that 
    have at least 1, 2, 3, 4, or 5 values greater than each value in the input array. The results 
    are displayed as a line plot.

    Parameters:
    ----------
    array : numpy.ndarray
        A 1D or 2D numpy array where each element represents a concept activation target (z). 
        If a 2D array is provided, the function evaluates each row for activations above the 
        threshold values specified by the input array.

    Returns:
    -------
    None
        This function does not return any value. It generates a plot directly.

    Example:
    --------
    import numpy as np
    
    # Create a sample 2D array of activations
    activations = np.random.rand(100, 10)  # 100 samples with 10 activations each
    
    # Call the plotting function
    plot_percentage_values(activations)

    Notes:
    ------
    - The function assumes that the input array is a numpy array. Ensure that numpy and matplotlib 
      are imported in your environment.
    - The plot displays percentages on the y-axis and the concept activation targets on the x-axis,
      making it easy to visualize the distribution of activation values across different thresholds.

    """
    percent_one_value = []
    percent_two_values = []
    percent_three_values = []
    percent_four_values = []
    percent_five_values = []

    for x in array:
        count_one_value = np.sum(np.any(array > x, axis=1))
        percent_one_value.append((count_one_value / len(array)) * 100)
        
        count_two_values = np.sum(np.sum(array > x, axis=1) >= 2)
        percent_two_values.append((count_two_values / len(array)) * 100)

        count_three_values = np.sum(np.sum(array > x, axis=1) >= 3)
        percent_three_values.append((count_three_values / len(array)) * 100)

        count_four_values = np.sum(np.sum(array > x, axis=1) >= 4)
        percent_four_values.append((count_four_values / len(array)) * 100)

        count_five_values = np.sum(np.sum(array > x, axis=1) >= 5)
        percent_five_values.append((count_five_values / len(array)) * 100)

    plt.figure(figsize=(10, 6))
    plt.plot(array, percent_one_value, label='at least 1 activations > z', color='blue', linestyle='-', linewidth=2)
    plt.plot(array, percent_two_values, label='at least 2 activations > z', color='red', linestyle='-', linewidth=2)
    plt.plot(array, percent_three_values, label='at least 3 activations > z', color='green', linestyle='-', linewidth=2)
    plt.plot(array, percent_four_values, label='at least 4 activations > z', color='orange', linestyle='-', linewidth=2)
    plt.plot(array, percent_five_values, label='at least 5 activations > z', color='black', linestyle='-', linewidth=2)
    plt.xlabel('Concept Activation Target (z)')
    plt.ylabel('Percentage of points (%)')
    plt.title('Percentage of Points with at least N values > z in Concept Activation Targets')
    plt.legend()
    plt.grid(True)
    plt.show()


def plotSpectrogram(ax, spectrogram, title, vmax, vmin, denorm=None, cut=None):
    """
    Plots a single spectrogram on a given axis.

    This function takes a spectrogram array and visualizes it using a heatmap. It allows for 
    denormalization of the spectrogram data and can truncate the time axis if specified. 

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the spectrogram. This allows for multiple spectrograms to be 
        plotted in subplots.

    spectrogram : numpy.ndarray
        A 2D array representing the spectrogram data. One axis typically represents frequency 
        and the other represents time. The input should be in a format compatible with heatmap 
        visualization.

    title : str
        The title for the spectrogram plot. This will be displayed above the heatmap for context.

    vmax : float, optional
        The maximum value for the color scale. It defines the upper limit of the colormap used 
        for visualizing the spectrogram.

    vmin : float, optional
        The minimum value for the color scale. It defines the lower limit of the colormap used 
        for visualizing the spectrogram.

    denorm : tuple of (float, float), optional
        A tuple containing the mean and standard deviation to denormalize the spectrogram data. 
        If provided, the function will adjust the spectrogram values back to their original scale.

    cut : int, optional
        An integer specifying the number of time frames to keep in the spectrogram. If provided, 
        it will truncate the spectrogram to only show data for the first `cut` time frames.

    Returns:
    -------
    None
        This function does not return any value. It generates a plot directly on the provided axis.

    Example:
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create a new figure and axis
    fig, ax = plt.subplots()
    
    # Create a sample spectrogram data
    spectrogram = np.random.rand(100, 200)  # Example 2D array for frequencies and time
    
    # Call the plotting function
    plotSpectrogram(ax, spectrogram, title='Sample Spectrogram', vmax=1, vmin=0)

    Notes:
    ------
    - The function uses the `seaborn` library to create the heatmap. Ensure that `seaborn` is 
      installed in your environment.
    - The y-axis is inverted by default, which is typical for spectrogram representations, 
      where lower frequencies are usually displayed at the bottom.

    """
    spectrogram = spectrogram.numpy()
    if (denorm is not None):
        mean, std = denorm
        spectrogram = spectrogram * std + mean
        spectrogram = 10**(spectrogram/20)
        cmap='Blues'
        vmax = None
        vmin = None
    else:
        cmap='coolwarm'

    if cut is not None:
        spectrogram = spectrogram[:, : cut]

    spectrogram = spectrogram.transpose(1, 0)
   
    sns.heatmap(spectrogram, cmap=cmap, cbar=True, vmax=vmax, vmin=vmin, ax=ax,)

    ax.set_title(title)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.invert_yaxis()

def plotSpectrograms(spectrograms, titles, vmax, vmin, denorm=None, cut = None):
    """
    Plots multiple spectrograms in a grid format.

    This function takes a list of spectrograms and visualizes them in a series of subplots, 
    allowing for comparative analysis of different spectrograms. Each spectrogram is plotted 
    with a specific color scale defined by the `vmax` and `vmin` parameters.

    Parameters:
    ----------
    spectrograms : list of numpy.ndarray
        A list containing the spectrogram data to be plotted. Each spectrogram should be a 
        2D array where one axis represents time and the other represents frequency.

    titles : list of str
        A list of titles corresponding to each spectrogram. The titles will be displayed above 
        each subplot for clarity.

    vmax : float
        The maximum value for the color scale. It defines the upper limit of the colormap 
        used for visualizing the spectrograms.

    vmin : float
        The minimum value for the color scale. It defines the lower limit of the colormap 
        used for visualizing the spectrograms.

    denorm : function, optional
        A function to denormalize the spectrogram data if it has been normalized. This allows 
        for the restoration of the original scale of the data for accurate visualization.

    cut : float, optional
        A value to limit the display range of the spectrograms. If provided, it will truncate 
        values above the specified cut-off, which can be useful for focusing on specific 
        frequency ranges.

    Returns:
    -------
    None
        This function does not return any value. It generates plots directly to the output.

    Example:
    --------
    import numpy as np
    import matplotlib.pyplot as plt

    # Create some sample spectrogram data
    spectrogram1 = np.random.rand(100, 100)
    spectrogram2 = np.random.rand(100, 100)
    spectrograms = [spectrogram1, spectrogram2]
    titles = ['Spectrogram 1', 'Spectrogram 2']
    
    # Call the plotting function
    plotSpectrograms(spectrograms, titles, vmax=1, vmin=0)

    Notes:
    ------
    - The function assumes that the `plotSpectrogram` function is defined elsewhere in your code,
      which handles the individual plotting of each spectrogram.
    - The layout of the subplots is dynamically adjusted based on the number of spectrograms,
      ensuring that they fit well within the specified figure size.

    """
    n = len(spectrograms)
    fig, axes = plt.subplots(math.ceil(n/3), 3, figsize=(15, 5*math.ceil(n/3)))
    # Esegui il plot per ogni spettrogramma
    for i, (ax, spectrogram) in enumerate(zip(axes.flatten(), spectrograms)):
        plotSpectrogram(ax, spectrogram, titles[i], vmax=vmax, vmin=vmin, denorm=denorm, cut=cut)

    plt.tight_layout()
    plt.show()

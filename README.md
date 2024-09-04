# K-Means_Clustering Case

This project implements the K-Means clustering algorithm to group companies into different clusters based on their industry. This is a code that support an article published in LinkedIn.
The code is written in Python and uses libraries such as pandas, scikit-learn, matplotlib, and seaborn.

Requirements
Python 3.x
pandas
scikit-learn
matplotlib
seaborn
numpy
You can install the required libraries using pip:

pip install pandas scikit-learn matplotlib seaborn numpy

Usage
Load the data: The code loads the data from a CSV file named CompaniesProfiles_Scenario1.csv. Make sure the file is in the correct path. ("c:\temp)
Preprocessing: The data is encoded using one-hot encoding for categorical features and normalized using StandardScaler.
Calculate WCSS: The Within-Cluster Sum of Squares (WCSS) is calculated for different values of K (number of clusters) and the elbow method is plotted to determine the optimal number of clusters.
Apply K-Means: The K-Means algorithm is applied with the optimal number of clusters determined in the previous step.
Visualization: The clusters are visualized using scatter plots and bubble charts.
Execution
To run the code, simply clone the repository and execute the main script.

Play with this script by changing the number of clusters, k value.

kmeans_final = KMeans(n_clusters=57, random_state=42)
and KMeans parameters.

The article explains two scenarios: k=5 and k=57.

This script creates one output file with a cluster assignment for each company.

Author
This code was created by Vicenç Luna Quintana

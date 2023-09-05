import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/gaga/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/e6b30b2c0d21a68493764c9db555b026/Message/MessageTemp/3a43f9b3cdb816b75ce4987aecdb71eb/File/optimizer_201901_07.csv')

# Convert the "ECU_SEND_LOCALTIME" column to datetime type
df['ECU_SEND_LOCALTIME'] = pd.to_datetime(df['ECU_SEND_LOCALTIME'])

# Group the DataFrame by "ECU_SEND_LOCALTIME"
groups = df.groupby('ECU_SEND_LOCALTIME')

# Create an empty DataFrame to store the outliers
outliers_df = pd.DataFrame()

model = "isolation_forest"
if model == "DBSCAN":
    # Iterate over the groups
    for group_id, (name, group) in enumerate(groups):
        # Select the columns for comparison
        data = group[['INPUT_VOLTAGE', 'INPUT_CURRENT', 'INPUT_POWER', 'INPUT_ENERGY']]

        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)

        # Perform clustering using DBSCAN algorithm on normalized data
        dbscan = DBSCAN(eps=2, min_samples=3)
        clusters = dbscan.fit_predict(normalized_data)

        # Add the cluster labels to the group DataFrame
        group['Cluster'] = clusters

        # Add the group ID column
        group['GroupID'] = group_id

        # Identify the outliers based on the cluster labels
        outliers = group[group['Cluster'] == -1]

        # Append the outliers to the outliers DataFrame
        outliers_df = outliers_df.append(outliers)

        print(outliers)
        print()

    # Save the outliers DataFrame to a CSV file
    outliers_df.to_csv('outliers_dbscan.csv', index=False)

elif model == "isolation_forest":
    # Iterate over each group
    for group_id, (name, group_data) in enumerate(groups):
        # Select the relevant columns for outlier detection
        X = group_data[['INPUT_VOLTAGE', 'INPUT_CURRENT', 'INPUT_POWER', 'INPUT_ENERGY']]

        # Normalize the data
        scaler = StandardScaler()
        normalized_X = scaler.fit_transform(X)

        # Perform outlier detection using Isolation Forest on normalized data
        isolation_forest = IsolationForest(contamination=0.01)
        isolation_forest.fit(normalized_X)
        outliers_indices = isolation_forest.predict(normalized_X) == -1

        # Add a column to distinguish each group
        group_data['GroupID'] = group_id

        # Append the outliers to the DataFrame
        outliers_df = outliers_df.append(group_data[outliers_indices])

        print(group_data[outliers_indices])
        print()

    # Save the outliers DataFrame to a CSV file
    outliers_df.to_csv('outliers_isolation_forest.csv', index=False)
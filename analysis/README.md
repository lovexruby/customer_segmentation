# Customer Segmentation with K-Means Clustering

This project performs customer segmentation using the K-Means clustering algorithm on a dataset of mall customers. The goal is to identify distinct customer groups based on their annual income and spending behavior.

## Project Structure

customer_segmentation/
├── data/
│ └── Mall_Customers.csv # Raw dataset
├── analysis/
│ └── segmentation.py # Python script for analysis and visualization
├── plots/
│ └── customer_segments.png # Resulting plot
├── .gitignore # Files/folders to exclude from Git
└── README.md # Project documentation


## Dataset Overview

- **Source**: [Kaggle - Mall Customer Segmentation](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Features used**:
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

These features are used to group customers into clusters based on their behavior.

## Tools and Libraries

- pandas
- scikit-learn
- matplotlib
- seaborn

## Methodology

1. Load and inspect the dataset
2. Select relevant features
3. Standardize the data using `StandardScaler`
4. Apply K-Means clustering (with 5 clusters)
5. Visualize the resulting clusters in a 2D scatter plot

## Results

The final output is a scatter plot showing five distinct customer segments, color-coded based on cluster assignment. This segmentation can help businesses better understand and target different customer groups.

![Customer Segments](plots/customer_segments.png)

## Next Steps

- Use the Elbow Method to determine the optimal number of clusters
- Add more features (e.g., Age, Gender) to improve segmentation
- Deploy the analysis as an interactive dashboard using tools like Streamlit

## Author

Ruben Mendes  
Data Analyst in training, focused on practical data projects and clean, insightful visualizations.

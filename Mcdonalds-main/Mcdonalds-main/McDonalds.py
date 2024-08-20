from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('mcdonalds.csv')

print(df.columns)
print(df.shape)
print(df.head(3))
# Extract the segmentation variables
segmentation_vars = df.iloc[:, :11]

# Convert YES/NO to binary (1/0)
segmentation_vars = segmentation_vars.apply(lambda x: x == "Yes").astype(int)

print(segmentation_vars.mean().round(2))

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(segmentation_vars)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()
print("Standard Deviations:", pca.singular_values_)
print("Proportion of Variance:", explained_variance)
print("Cumulative Proportion:", cumulative_variance)

loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=segmentation_vars.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
print(loadings_df.round(2))

# Plot perceptual map
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], color='grey', alpha=0.5)
for i in range(segmentation_vars.shape[1]):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.1, head_length=0.1, fc='k', ec='k')
    plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, segmentation_vars.columns[i], color='k')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Principal Components Analysis')
plt.grid()
plt.show()




from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
z = linkage([[x] for x in [1,4,6,7]])
fig, ax = plt.subplots(figsize=[6.4, 2.4])
dendrogram(z, color_threshold=1, no_labels=False, ax=ax)
ax.set_yticks([])
ax.set_xticklabels([1, 2, 3, 4])
ax.set_xlabel("i")
plt.tight_layout()
plt.savefig("../contents/img/generated/cluster_dendogram.pdf")
plt.show()
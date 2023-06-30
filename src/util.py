import knime.extension as knext


def is_nominal(column):
    # Filter nominal columns
    return column.ktype == knext.string() or column.ktype == knext.bool_()

#Input: "How to use the K-Means node in KNIME"
#
#{
#    "video_title": "K-Means Clustering in KNIME Analytics Platform",
#    "video_link": "https://www.youtube.com/watch?v=0GmJqF7ZQ6A",
#    "video_description": "In this video, we will learn how to use the K-Means node in KNIME Analytics Platform.\n\nK-Means clustering is a popular machine learning algorithm used for clustering similar data points together. It works by dividing a set of data points into a specified number of clusters, with each cluster having a defined center point.\n\nIn KNIME, the K-Means node allows you to perform K-Means clustering on a dataset. You can specify the number of clusters, the distance metric to use, and other options related to how the algorithm should be executed.\n\nIn this tutorial, we will walk through an example of using the K-Means node in KNIME to cluster a dataset of customer demographics. We will start by reading in the data, preprocessing it, and then using the K-Means node to cluster the data into three distinct groups. We will also visualize the results of the clustering using a scatter plot.\n\nBy the end of this video, you should have a good understanding of how to use the K-Means node in KNIME to perform clustering on your own datasets."
#} 

main_cat = knext.category(
    path="/community",
    level_id="kai",
    name="K-AI",
    description="",
    icon="icons/ml.svg",
)
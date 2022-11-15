# Cluster Stability

## Summary
This repo explores how using clustering, dimensionality reduction kmeans, svd and other clustering techniques on text data and then generating images using stable diffusion. This is a technique to visualize the learned kmeans centroids. The text/topics are used as a query to a large database of image prompts and images and the nearest neighbors are found. The image nearest to the centroid is used as a representative image for that centroid. Cl


### Motivation
- Kmeans is a popular algorithm to generate clusters in an unsupervised fashion. We will see how to use it on text data. Then I show how to use stable diffusion to generate images for each cluster.

### ideas
- given a list of topics for a cluster, spherically interpolate through the text prompts and generate topic frames with stable diffusion model
- interpolating between centroids
- maybe generate an unsupervised news real

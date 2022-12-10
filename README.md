# Cluster Stability

## Summary
This repo explores using clustering, dimensionality reduction. kmeans, svd and potentially other clustering techniques on text data and then generating images using stable diffusion. This is a technique to visualize the learned kmeans centroids. The text/topics are used as a query to a large database of image prompts and images and the nearest neighbors are found. The image nearest to the centroid is used as a representative image for that centroid. I dont expect this to be good right away, however I expect this can be used to identify holes in the cached images and suggest prompts.

### Motivation
- Kmeans is a popular algorithm to generate clusters in an unsupervised fashion. We will see how to use it on text data. Then I show how to use stable diffusion to generate images for each cluster.
- add LDA 
- BertTopic
### ideas
- given a list of topics for a cluster, spherically interpolate through the text prompts and generate topic frames with stable diffusion model
- interpolating between centroids
- maybe generate an unsupervised news real
- clustered image mosaics using stable diffusion infinity 


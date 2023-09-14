"""
a selective transfer learning method based on TrAdaBoost using samples collected from different time-domain 

For the issue of the assessment of heavy metal concentrations using remote sensing is sample-intensive, with expensive model development.We developed a transfer learning model (i.e., based on the TrAdaboost algorithm) to monitor heavy metal pollution using samples collected from different time-domain. Firstly, spectral indices of Landsat8 multispectral images, terrain, and other auxiliary data representing the spatial distribution of soil heavy metals and soil sample data were acquired in 2017 and 2019. Then, a selective transfer TrAdaboost algorithm based on traditional TrAdaboost and transfer learning theory was used for the retrieval of Cu concentrations in the topsoil. 
# The principle of our selective transfer learning method based on TrAdaBoost algorithm 
"""
Stage 1.The source domain instance weight slowly decreases until it reaches a fixed value (determined by cv); Get the model with the lowest average error from CV - determine the instance weight
Stage 2.The weights of all source domain instances remain unchanged, and the weights of target domain instances are updated normally (Adaboost.R2);
"""
Note:Only the prediction results generated in the second stage are stored and used to determine the output of the result model¡£

# Master Thesis 2021-2022
Quorum Sensing Modelling for Pseudomonas Aeruginosa. \
Supervisor and Coordinator: Prof. Johannes Müller

## Intro
Quorum Sensing is a cell density dependent cooperative behaviour found in bacteria, 
while Pseudomonas Aeruginosa is a drug-resistant pathogen that causes thousands of deaths in hospital settings.
My mathematical models explain and predict several dynamics at different scales using Ordinary Differential Equations (ODEs) and stochastic processes. 
I used experimental data and a bifurcation analysis to choose parameters. All the simulations were implemented in Python using standard numerical integration techniques.
One cool achievement was that I reproduced complex "switch-like" behavior where bacteria suddenly activate when reaching critical thresholds.
Demonstrated how the system can lose this complex switching in favor of simpler gradual activation under certain conditions.
The model results are consistent with empirical results.

## Abstract:
We aim in this study to simulate the cooperative behaviour of bacterium Pseudomonas aeruginosa, a particularly dangerous pathogen in hospital settings. This bacterium employs several cooperative behavioral strategies to regulate virulence. Among them is Quorum Sensing, where the bacteria act differently depending on the cell concentration, measured through some signalling molecules. We aim to explain, predict and reproduce Quorum Sensing through our models. In addition, we attempt to replicate the bi-stable behaviour observed in experimental data. The model we have developed explains several dynamics at different scales using Ordinary Differential Equations (ODEs) and stochastic processes. The global dynamics at the colony level are explained through a deterministic model, while a stochastic model predicts the local cell-level dynamics. With experimental data and bifurcation analysis, we chose a parameter set that could show bi-stable behaviour. We analysed the stochastic model by simulating the distribution of the Master equations resulting from the stochastic process. All the simulations were implemented in Python using standard numerical integration techniques. The model has been successful in explaining quorum sensing behaviour in agreement with experimental data.

This repository stores the code I worked on in my master thesis.
You need to run the \__main\__.py file.

## CODE STRUCTURE
STRUCTURE:
 - Import Packages.
 - Set the data (from outside sources). 
 - Fit the parameters for logistic growth of the cell colture. Unused later.
 - Fit the parameters to the data relating extracellular autoinducer, intracellular
 autoinducer, synthetase, cell population in an ODE model. The parameters are saved and used in later functions.
 - Have a corresponding stochastic model. To analyze it master equations are derived
 and are simulated.
 There are three possible simulations. Group_sensing deals with group sensing, in which 
 extracellular autoinducer Se changes in time. 
 To account for parameter heterogeneity, in Group_heter_K five types of cell "classes" are counted, each with their own parameter K. For simplicities sake, these "classes" are and remain of equal size, interacting only through the release of autoinducer.
 Self_sensing accounts for self-sensing, in which extracelluar autoinducer is fixed (it needs to be stated as a parameter). Self_heter_K is self_sensing but with 5 different cell classes with different K values.
 - The bifurcation function checks if with the current parameters one is the 
 theoretical bistability range. If it is, it gives the (theoretical)
 range of extracellular autoinducer Se such that a bi-stable mode exists. In any 
 case, the bifurcation parameter K is given.

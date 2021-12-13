# 24-787-Final-Project-LP-MS
Final Project Code for Lauren Parola &amp; Mikayla Schneider for 24-787
--Data_Processing_Walking_Events.py--
This script contains the intial steps of syncing, filtering, and aligning the data from right and left thigh and shank IMUS. 
Then the periods of walking are identified, extracted as new "subjects", and exported to Excel to calculate features. 

--Temporal_Parameters.py--
This script contains the process of identifying each gait cycle via heel contact and toe-off points. 
From there, values of limp, gait cycle time, double-support, and swing time are calculated. 

--ML_Model_Application.py--
This script contains the final application of various ML classifiers and subsequent methods of reducing features for comparable accuracy. 

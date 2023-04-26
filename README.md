# VELOX_CTH

A algorithm to calculate the cloud top temperature (CTH) by linearly relating cloud top brightness temperature (BT) with dropsonde profiles T(z). This method is known from satellite CTH retrivals with generally high uncertainties. A method for calculating shallow cumuli CTH is proposed, that has small bias compared to HALO's onboard 532nm depol LIDAR WALES.   
To get started and explore the code, two notebooks are provided. For the basis calculation of lapse rate coefficients from dropsondes, and some infromation on VELOX' cloud mask and a 1d CTH calculatation, take a look at `CTH_coffs_notebook.ipynb`.   
The 2D-Version for CTH retrieval is shown in `CTH_fields_notebook.ipynb`, where the problem is extended into the plane.

## TODO

-   increase performance of CTH_calculate_field.py, a julia version would be nice
-   add the tutorials from CTH_field_notebook.ipynb, CTH_coffs_notebook.ipynb to README.md
-   fork bt catalogue to eurc4a repo

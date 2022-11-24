# VELOX_CTH

A algorithm to calculate the cloud top temperature (CTH) by linearly relating cloud top brightness temperature (BT) with dropsonde profiles T(z). This method is known from satellite CTH retrivals (source!) with generally high uncertainties. A method for calculating shallow cumuli CTH is proposed, that has small bias compared to HALO's onboard 532nm depol LiDaR WALES.  

## TODO

-   increase performance of CTH_calculate_field.py, a julia version would be nice
-   add the tutorials from CTH_field_notebook.ipynb, CTH_coffs_notebook.ipynb to README.md
-   fork bt catalogue to eurc4a repo

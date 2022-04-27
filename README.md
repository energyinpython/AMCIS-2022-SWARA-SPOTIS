# AMCIS-2022-SWARA-SPOTIS

The Methodological Framework for Temporal SWARA-SPOTIS Sustainability Assessment.

This Python 3 project includes methods implemented for multi-criteria temporal assessment with the SPOTIS method, MEREC criteria weighting method, 
and SWARA weighting method for determining considered periods' significance.

## Methods

- the SPOTIS method

- Objective methods for criteria weighting:

	- Equal weighting `equal_weighting`
	- Entropy weighting `entropy_weighting`
	- Standard deviation weighting `std_weighting`
	- CRITIC weighting `critic_weighting`
	- Gini coefficient-based weighting `gini_weighting`
	- MEREC weighting `merec`
	- Statistical variance weighting `stat_variance_weighting`
	
- Subjective weighting method SWARA `swara_weighting` for determining significance of periods'

- Normalization methods for decision matrix normalization:

	- Linear normalization `linear_normalization`
	- Minimum-Maximum normalization `minmax_normalization`
	- Maximum normalization `max_normalization`
	- Sum normalization `sum_normalization`
	- Vector normalization `vector_normalization`
	
- Correlation coefficients:

	- Spearman Rank Correlation coefficient `spearman`
	- Weighted Spearman Rank Correlation coefficient `weighted_spearman`
	- Rank Similarity coefficient `coeff_WS`
	- Pearson Correlation coefficient `pearson_coeff`
	- Kendall coefficient `kendall`
	- Goodman-Kruskal coefficient `goodman_kruskal`
	
- Method for ranking alternatives according to MCDA preference values `rank_preferences`

## License

This project is licensed under the terms of the MIT license.
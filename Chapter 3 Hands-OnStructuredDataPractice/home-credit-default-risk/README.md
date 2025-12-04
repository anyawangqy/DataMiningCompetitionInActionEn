
The code for this project is modified based on NoxMoon's project home-credit-default-risk (https://github.com/NoxMoon/home-credit-default-risk).

## Competition Introduction
To ensure that people obtain loans responsibly, Home Credit aims to use various data (including telecommunications and transaction information) to predict clients' repayment abilities.
Specifically, participants need to synthesize various information to predict whether each loan will be repaid late.



## Overall Solution
<img src="framework.png" width = "1500" alt="logo" align=center />


## data
Download the data to the input folder and unzip it.
- Data download address: https://www.kaggle.com/competitions/home-credit-default-risk/data

## Code File Description

```
src
├── ensemble
	├── opt_weights.py # Ensemble learning, weighted fusion of different single-model results
├── feature_engineer
	├── house-doc-feats.py
	# Main table meta feature construction
	├── cc-ts.py
	# Construct 1-to-many subsidiary table temporal meta features for the credit_card_balance table
	├── pos-ts.py
	# Construct 1-to-many subsidiary table temporal meta features for the POS_CASH_balance table
	├── bubl-ts.py
	# Construct 1-to-many subsidiary table temporal meta features for the bureau table
	├── inst-ts.py
	# Construct 1-to-many subsidiary table temporal meta features for the installments_payments table
	├── prev-training.py
	# Construct 1-to-many subsidiary table meta features for the previous_application table
	├── buro-training.py
	# Construct 1-to-many subsidiary table meta features for the bureau table
	└── month-training.py
	# Aggregate the POS_CASH_balance table by SK_ID_CURR and month, calculate aggregated statistical information for other columns to create a new 1-to-many subsidiary table, and then generate subsidiary table meta features for this new table
└── model
	├── lgb1.py
	# lightgbm single model, first set of hyperparameters
    	Includes other features (business features, categorical feature encoding, aggregated statistical features, etc.)
    	And downsampling strategy
	├── lgb2.py # Same as above, second set of hyperparameters
	├── lgb3.py # Same as above, third set of hyperparameters
	└── xgb1.py # Same as above, using the xgboost single model
```
## Execution Method
1. Execute feature engineering

```
python ./src/feature_engineer/prev-training.py
python ./src/feature_engineer/buro-training.py
python ./src/feature_engineer/month-training.py
python ./src/feature_engineer/house-doc-feats.py
python ./src/feature_engineer/inst-ts.py
python ./src/feature_engineer/bubl-ts.py
python ./src/feature_engineer/pos-ts.py
python ./src/feature_engineer/cc-ts.py
```
2. Execute the models
```
python ./src/model/lgb1.py
python ./src/model/lgb2.py
python ./src/model/lgb3.py
```
3. Execute ensemble learning
```
python ./src/ensemble/opt_weights.py
```
Results are saved in the output folder (the final file is ensemble.csv)


## Final Results

|       |  Private LB | Public LB  |
|-------| ------------ | ------------ |
| lgb_1 |  0.80026 | 0.80491  |
| lgb_2 | 0.80021 | 0.80492  |
| lgb_3 | 0.80011 | 0.80476  |
| ensemble     | 0.80028| 0.80494 |

Final result ranking:36/7176
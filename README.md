# Lymphoma classification
[![](https://zenodo.org/badge/DOI/10.5281/zenodo.5637324.svg)](https://doi.org/10.5281/zenodo.5637324)
[![](https://img.shields.io/github/release/mbarbetti/lymphoma-classification.svg)](https://GitHub.com/mbarbetti/lymphoma-classification/releases/)
[![](https://img.shields.io/github/issues/mbarbetti/lymphoma-classification.svg)](https://github.com/mbarbetti/lymphoma-classification/issues/)
[![](https://img.shields.io/github/issues-pr/mbarbetti/lymphoma-classification.svg)](https://github.com/mbarbetti/lymphoma-classification/pulls/)
[![](https://badgen.net/github/stars/mbarbetti/lymphoma-classification)](https://github.com/mbarbetti/lymphoma-classification/stargazers/)

## About
Classification of bulky mediastinal lymphomas based on radiomic features.

## Jupyter notebooks
[**`0_dataset_cleaning.ipynb`**](https://github.com/mbarbetti/lymphoma-classification/blob/master/0_dataset_cleaning.ipynb)
  - Dataset cleaning
  - Data-format correction
  - **NEW!** Treatment of NaN values

[**`1_dataset_visualization.ipynb`**](https://github.com/mbarbetti/lymphoma-classification/blob/master/1_dataset_visualization.ipynb)
  - Visualization of the radiomic features distributions
  - **NEW!** Systematic study of LIFEx output (based on [LIFEx docs](https://www.lifexsoft.org/index.php/resources/documentation))
  - Heavy reduction of input features to remove duplicated variables
  - **NEW!** Only robust and independent texture features [[Orlhac 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145063)] are kept

[**`2_dataset_reduction.ipynb`**](https://github.com/mbarbetti/lymphoma-classification/blob/master/2_dataset_reduction.ipynb)
  - Correlation studies
  - Removal of strongly-correlated features
  - **NEW!** Use of U-test to measure the discriminant power
  - Removal of features without discriminant power

[**`3_binary_classification.ipynb`**](https://github.com/mbarbetti/lymphoma-classification/blob/master/3_binary_classification.ipynb)
  - Binary classification of lymphomas in cHL and PMBCL
  - Models tested: logistic regression, random forest and gradient BDT
  - **NEW!** More models tested: linear SVM and gaussian processes
  - Comparison of models performance (ROC curves)
  - Evaluation of models performance on test-set
  - **NEW!** Uncertainties study (only on [`model_bin_classifier.py`](https://github.com/mbarbetti/lymphoma-classification/blob/master/model_bin_classifier.py))

[**`4_multi_classification.ipynb`**](https://github.com/mbarbetti/lymphoma-classification/blob/master/4_multi_classification.ipynb)
  - Binary classifiers promoted to multiclass classifiers
  - Multiclass classification of lymphomas in cHL, GZL and PMBCL
  - **NEW!** More models tested: linear SVM and gaussian processes
  - **NEW!** Comparison of models performance (one-vs-all ROC curves)
  - Evaluation of models performance on test-set
  - **NEW!** Uncertainties study (only on [`model_multi_classifier.py`](https://github.com/mbarbetti/lymphoma-classification/blob/master/model_multi_classifier.py))

## License

[MIT License](LICENSE)

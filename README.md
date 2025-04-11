<h1 align="center">A Comparison of Supervised Learning Algorithms</h1>

<p align="center">
<b>Aydin Tabatabai</b><br>
University of California, San Diego <br> 
atabatabai@ucsd.edu
</p>

<h2 align="center">Abstract</h2>

Supervised learning algorithms have been widely applied across various domains, but their performance often depends on dataset characteristics and evaluation methods. This report presents a comparison of three supervised learning classifiers: Random Forest, Support Vector Machine (SVM), and Logistic Regression. These classifiers were evaluated on three datasets, with each classifier being tested using three different train/test splits (20/80, 50/50, and 80/20) and tuned to optimize their hyperparameters. The performance of the models was assessed using metrics such as training, validation, and testing accuracy. The results highlight the importance of dataset characteristics and hyperparameter tuning in determining the effectiveness of classifiers.

## Introduction

Machine learning algorithms have become essential tools for solving complex problems across different fields, including healthcare, finance, and engineering. Among these, supervised learning methods are highly used for classification tasks. Evaluating the performance of supervised learning classifiers is critical to understanding their effectiveness and identifying the conditions under which they perform best.

This paper focuses on comparing the performance of three commonly used classifiers: Random Forest, Support Vector Machine (SVM), and Logistic Regression. These algorithms were chosen due to their popularity and unique approaches to classification. Three datasets from the UCI Machine Learning Repository were used to evaluate these classifiers: Raisin \[1\], Early Stage Diabetes Risk Prediction \[2\], and Car Evaluation \[3\]. These datasets span different domains and contain diverse feature types, allowing for a thorough analysis of the algorithm performance. Each classifier was evaluated on training, validation, and testing accuracies using three different splits of training and testing data which are 20/80, 50/50, and 80/20. Hyperparameter tuning with cross-validation was also performed to identify the best configurations for each model.

The purpose of this study is to provide insights into the relative performance of these classifiers and to explore how factors such as dataset characteristics, hyperparameter settings, and train/test splits influence the classification accuracy. The results aim to highlight the strengths and limitations of each algorithm under varying conditions and provide insight into use cases of specific algorithms for real-world situations.

## Method

### Classifiers  
This study focused on three supervised learning classifiers, Random Forest, Support Vector Machine (SVM), and Logistic Regression, each offering a useful approach to classification. Random Forest is a learning method that combines multiple decision trees to improve performance. The key hyperparameter tuned for Random Forest was maximum depth. SVM is a kernel-based method that finds the optimal hyperplane separating classes in feature space. The key hyperparameter tuned for SVM was the C parameter, the trade-off parameter that determines the strength of the regularization. Logistic Regression, predicts probabilities for classification. The key hyperparameter tuned for SVM was the C parameter, which controls the trade-off between a simple model with a larger margin and a complex model that fits the training data more closely. Scikit-learn 1.4.2 was used for all of the classifiers and hyperparameter tuning.

### Datasets
Three datasets from the UCI Machine Learning Repository were used to train and test the classifiers. The first dataset, Raisin, has seven numerical features representing the properties of raisins with two classes, Kecimen and Besni. The second dataset, Early Stage Diabetes Risk Prediction, has a mix of numerical and categorical features describing symptoms and demographic factors, with two classes: Positive and Negative. The third dataset, Car Evaluation, has six categorical features describing car characteristics. The original classes (unacc, acc, good, vgood) were merged into two categories for binary classification, with unacc as the negative class and the rest as the positive class.

### Data Preprocessing
Preprocessing was applied to prepare the datasets for analysis. Categorical features in the Early Stage Diabetes and Car Evaluation datasets were encoded numerically using LabelEncoder. Numerical features in all datasets were standardized using StandardScaler to ensure consistent scaling. For the Car Evaluation dataset, the multiple classes were transformed into binary labels. Across all datasets, classes were encoded numerically, with 0 representing negative class and 1 representing positive class.

### Hyperparameter Tuning
Hyperparameter tuning was conducted using GridSearchCV to optimize the classifiers. For Random Forest, the maximum depth hyperparameter was tuned with values \[10, 20, 30\]. For SVM, the parameter C was tuned with values \[0.1, 1, 10, 100\] while using the RBF kernel. Logistic Regression also was tuned with values \[0.1, 1, 10, 100\] for the C parameter. 

### Evaluation Metrics
The classifiers were evaluated using three train/test splits(20/80, 50/50, and 80/20) to analyze how training data size affects performance. Metrics included training accuracy, validation accuracy, and testing accuracy. These metrics provided a comprehensive view of each classifier’s performance by evaluating how well the models fit the training data, how effectively they generalized to unseen validation data during tuning, and how accurately they performed on the final test data.

## Experiment

To evaluate the performance of the three classifiers (Random Forest, SVM, and Logistic Regression) experiments were conducted on three datasets, Raisin, Early Stage Diabetes Risk Prediction, and Car Evaluation. Each dataset was split into three train/test configurations: 20/80, 50/50, and 80/20, to analyze the impact of training size on performance. The metrics used for evaluation were training accuracy, validation accuracy, and test accuracy, with hyperparameter tuning.

The Raisin Dataset consists of 900 instances with 7 numerical features describing the different properties of raisins. Random Forest achieved perfect training accuracy of 1.000 for both the 20/80 and 80/20 splits, but its testing accuracy varied slightly, with a maximum of 0.867 for the 80/20 split using a maximum depth of 20\. SVM improved with more training data, achieving a peak testing accuracy of 0.872 with a C-value of 0.1 for the 80/20 split. Logistic Regression showed improvement as the training size increased, reaching its best testing accuracy of 0.889 at the 80/20 split with a C-value of 100\. Overall, all the classifiers performed reasonably well with test accuracy, and Random Forest performed the best with training accuracy.

<p align="center">
<i>Table 1: Raisin Dataset (900 instances, 7 feature)</i>
</p>

| Model | Train Percentage | Training Accuracy | Validation Accuracy | Test Accuracy | Max Depth Hyperparameter | C-Value Hyperparameter |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Random Forest | 20% | 1.000 | 0.839 | 0.846 | 10 | N/A |
| Random Forest | 50% | 0.998 | 0.873 | 0.831 | 10 | N/A |
| Random Forest | 80% | 1.000 | 0.852 | 0.867 | 20 | N/A |
| SVM | 20% | 0.867 | 0.867 | 0.857 | N/A | 1 |
| SVM | 50% | 0.887 | 0.880 | 0.848 | N/A | 0.1 |
| SVM | 80% | 0.869 | 0.865 | 0.872 | N/A | 0.1 |
| Logistic Regression | 20% | 0.878 | 0.878 | 0.864 | N/A | 100 |
| Logistic Regression | 50% | 0.882 | 0.871 | 0.853 | N/A | 0.1 |
| Logistic Regression | 80% | 0.863 | 0.867 | 0.889 | N/A | 100 |

The Early Stage Diabetes Risk Prediction Dataset contains 520 instances with 16 features representing patient symptoms and demographics. Random Forest achieved perfect training accuracy of 1.000 for all splits, with testing accuracy peaking at 0.990 for the 80/20 split using a maximum depth of 20\. SVM consistently performed well, reaching a testing accuracy of 0.971 at the 80/20 split with a C-value of 10\. Logistic Regression, while performing slightly worse than the other classifiers, achieved its best testing accuracy of 0.942 at the 80/20 split with a C-value of 100\. For this dataset, Random Forest and SVM both showed strong generalization capabilities, while Logistic Regression was slightly less effective in modeling the relationships within the data.

<p align="center">
<i>Table 2: Early Stage Diabetes Risk Prediction Dataset (520 instances, 16 features)</i>
</p>

| Model | Train Percentage | Training Accuracy | Validation Accuracy | Test Accuracy | Max Depth Hyperparameter | C-Value Hyperparameter |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Random Forest | 20% | 1.000  | 0.894  | 0.945 | 10 | N/A |
| Random Forest | 50% | 1.000  | 0.958  | 0.950 | 10 | N/A |
| Random Forest | 80% | 1.000  | 0.966 | 0.990 | 20 | N/A |
| SVM | 20% | 0.990 | 0.913 | 0.938 | N/A | 10 |
| SVM | 50% | 0.985 | 0.946 | 0.962 | N/A | 1 |
| SVM | 80% | 0.995 | 0.957 | 0.971 | N/A | 10 |
| Logistic Regression | 20% | 0.942 | 0.913  | 0.889  | N/A | 1 |
| Logistic Regression | 50% | 0.950 | 0.919 | 0.892 | N/A | 10 |
| Logistic Regression | 80% | 0.940  | 0.918 | 0.942  | N/A | 100 |

The Car Evaluation Dataset includes 1,728 instances and 6 categorical features related to car characteristics. Random Forest once again achieved perfect training accuracy of 1.000 across all splits, with testing accuracy reaching a maximum of 0.991 at the 80/20 split using a maximum depth of 10\. SVM performed exceptionally well, achieving perfect testing accuracy of 1.000 at the 80/20 split with a C-value of 100\. Logistic Regression, on the other hand, performed significantly worse on this dataset, with testing accuracies ranging from 0.697 to 0.716. This suggests that Logistic Regression struggled with the categorical nature of the features, while SVM and Random Forest effectively captured the underlying patterns in the data.

<p align="center">
<i>Table 3: Car Evaluation Dataset (1728 instances, 6 features)</i>
</p>

| Model | Train Percentage | Training Accuracy | Validation Accuracy | Test Accuracy | Max Depth Hyperparameter | C-Value Hyperparameter |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Random Forest | 20% | 1.000 | 0.942 | 0.952 | 10 | N/A |
| Random Forest | 50% | 1.000  | 0.969 | 0.985 | 20 | N/A |
| Random Forest | 80% | 1.000  | 0.988 | 0.991 | 10 | N/A |
| SVM | 20% | 0.997 | 0.901 | 0.962 | N/A | 10 |
| SVM | 50% | 1.000 | 0.976 | 0.988 | N/A | 100 |
| SVM | 80% | 1.000 | 0.992 | 1.000 | N/A | 100 |
| Logistic Regression | 20% | 0.754  | 0.730 | 0.716 | N/A | 1 |
| Logistic Regression | 50% | 0.718 | 0.713 | 0.697 | N/A | 10 |
| Logistic Regression | 80% | 0.719 | 0.711 | 0.702 | N/A | 1 |

Overall, the results across all three datasets highlight several key trends. Random Forest consistently achieved high performance but showed signs of overfitting, particularly with smaller training sizes. SVM demonstrated robust performance across all datasets and train/test splits, proving to be a strong choice for both numerical and categorical data. Logistic Regression, struggled on more complex datasets. The experiments also revealed that increasing the training size improved test accuracy, as seen with the 80/20 split consistently yielding the best results. Hyperparameter tuning played a critical role in model performance by optimizing the key parameters for each classifier, allowing the classifiers to achieve the right balance between underfitting and overfitting to generalize well.

## Conclusion

This project evaluated the performance of three supervised learning classifiers, Random Forest, Support Vector Machine, and Logistic Regression, on three datasets, Raisin, Early Stage Diabetes Risk Prediction, and Car Evaluation. The classifiers were assessed using three train/test splits (20/80, 50/50, and 80/20) and hyperparameter tuning was performed to optimize their performance. Metrics such as training accuracy, validation accuracy, and testing accuracy were used to analyze and compare their effectiveness.

The results demonstrated that Random Forest and SVM consistently outperformed Logistic Regression across all datasets. Random Forest achieved perfect training accuracy in most cases, but it showed signs of overfitting, particularly with smaller training splits like 20/80. Despite this, it remained a strong performer, often achieving high testing accuracy, especially when sufficient training data was available. SVM displayed robust generalization across all datasets, with testing accuracies improving as the training size increased. It was particularly effective for datasets with complex patterns, such as Car Evaluation and Early Stage Diabetes Risk Prediction. Logistic Regression, struggled on more complex datasets and performed worse, particularly with smaller training splits. However, its performance improved with larger training sets, as seen in the Raisin and Early Stage Diabetes datasets.

The impact of training size was consistent across all classifiers and datasets. Larger training sets, such as the 80/20 split, consistently led to higher testing accuracy, illustrating the importance of data size and availability for achieving better generalization. Hyperparameter tuning also played a role in improving performance, where selecting the appropriate C parameter and tuning maximum depth significantly influenced and benefited the results.

In conclusion, Random Forest and SVM emerged as the most effective classifiers in this study, with SVM demonstrating the best overall balance between accuracy and generalization. Logistic Regression, though less effective on complex datasets, provided reasonable performance. These results demonstrate the importance of dataset characteristics, hyperparameter tuning, and training size when selecting machine learning classifiers for classification tasks. Future work could explore different classifiers, multiple class classification problems, and more advanced techniques for handling different data characteristics.

## References
\[1\] Çinar, İ., Koklu, M., & Tasdemir, S. (2020). Raisin \[Dataset\]. UCI Machine Learning  
Repository. https://doi.org/10.24432/C5660T.

\[2\] Early Stage Diabetes Risk Prediction \[Dataset\]. (2020). UCI Machine Learning Repository.  
https://doi.org/10.24432/C5VG8H.

\[3\] Bohanec, M. (1988). Car Evaluation \[Dataset\]. UCI Machine Learning Repository.  
https://doi.org/10.24432/C5JP48.
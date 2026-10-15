# Literature Review — Credit Scoring Benchmarking and Hybrid Ensemble

> Source: *A Comparative Benchmarking and Hybrid Ensemble Framework for Credit Scoring*, Capstone Project I — Final Report, October 2026 draft, Section 2 and References.
>
> This document summarises the literature relevant to the project's modelling framework. The references follow IEEE formatting and are provisionally numbered to match the Final Report.

## Overview

This literature review examines research published from 2020 onwards, focusing on five areas relevant to the project: credit-scoring model selection, credit data and feature engineering, ensemble learning, probability calibration, and explainability. Rather than summarising individual studies in isolation, it synthesises findings that inform the design and evaluation of the proposed credit-scoring framework.

Existing research demonstrates that predictive performance alone is insufficient for evaluating credit-risk models. Data characteristics, class imbalance, probability reliability, interpretability, and practical financial applications also influence their usefulness. The project does not claim to introduce new algorithms or techniques. Instead, it aims to provide a controlled comparison of established methods and integrate selected components into a reproducible credit-risk decision-support workflow.

## Credit Scoring and Model Selection

Recent credit-scoring research highlights the increasing use of machine learning alongside traditional statistical approaches. Dastile et al. [1] reviewed statistical and machine-learning methods, demonstrating the continuing relevance of conventional approaches while recognising the growing importance of more flexible predictive models. Similarly, Shi et al. [2], Noriega et al. [3], Zhang and Yu [4], and Montevechi et al. [5] examine developments in credit-risk modelling, including statistical methods, tree-based learners, neural networks, and ensemble techniques. Collectively, these studies support comparative evaluation rather than assuming that a particular modelling family consistently outperforms alternatives.

The present project therefore evaluates three distinct base models: Logistic Regression (LR), XGBoost, and a multilayer perceptron (MLP). Logistic Regression provides an interpretable statistical benchmark, XGBoost represents nonlinear gradient-boosted decision trees, and MLP introduces a neural-network approach.

Gunnarsson et al. [6] investigated deep learning for credit scoring, highlighting the importance of evaluating neural-network approaches against established alternatives. Liu et al. [7] explored the application of machine learning to default-probability prediction, while Alonso Robisco and Carbó Martínez [8] examined model-risk-adjusted performance in credit-default prediction. Together, these studies indicate that model selection should consider both predictive performance and practical evaluation requirements.

Accordingly, the MLP is included not because it is expected to outperform XGBoost, but because it provides a different modelling architecture whose predictions may offer complementary information within the stacking framework.

## Credit Data, Feature Engineering and Class Imbalance

Credit-scoring datasets often contain heterogeneous financial and demographic variables, with substantially fewer default cases than non-default cases. This imbalance creates challenges for model evaluation and may affect the trade-off between identifying defaults and producing reliable probability estimates.

Lenka et al. [9] reviewed ensemble-learning approaches for imbalanced credit-scoring datasets, while Hussin Adam Khatir and Bee [10] investigated combinations of machine-learning models and data-balancing techniques. Abedin et al. [11] examined weighted SMOTE combined with ensemble learning for predicting small-business credit risk. These studies demonstrate the relevance of imbalance-aware modelling but do not establish that any single balancing method is universally preferable.

For the present project, class-imbalance treatments were evaluated empirically using validation data that preserved the natural outcome distribution. Rather than retaining a method solely because it improved a particular classification metric, the team considered its overall effect on model performance and probability estimates.

Feature engineering was also evaluated as an experimental component. Zhang and Yu [4] discuss the importance of data characteristics in credit-risk assessment, Yang et al. [12] examine feature enhancement within a hybrid ensemble, and Hlongwane et al. [13] investigate the potential contribution of alternative data to credit scoring. These studies support the broader principle that the representation and selection of borrower information can influence predictive performance.

Within the project's experimental workflow, employment normalisation, logarithmic transformations of skewed monetary variables, and financial ratios were investigated. Employment normalisation did not provide a meaningful validation improvement, while logarithmic transformations produced modest, model-dependent changes. Financial ratios provided the strongest improvement among the tested feature-engineering strategies, particularly for XGBoost.

These findings reinforce the importance of selecting feature transformations according to validation evidence rather than assuming that additional engineered features will necessarily improve performance.

## Ensemble Learning and Stacking

Ensemble methods combine multiple predictive models to exploit potentially complementary patterns in the data. Lenka et al. [9] discuss ensemble learning in imbalanced credit scoring, Zou et al. [14] investigate a bagging-cascading boosted decision-tree approach, and Yang et al. [12] propose a hybrid credit-scoring ensemble incorporating feature enhancement and optimised soft voting.

Together, these studies provide a rationale for evaluating model combinations while also demonstrating that ensemble performance depends on the underlying learners and combination strategy.

Stacking differs from fixed averaging or voting by training a meta-model to combine base-model predictions. Yin et al. [15] applied stacking to personal credit-risk assessment in peer-to-peer lending, providing a relevant example of its use in financial prediction.

In the present project, Logistic Regression, XGBoost, and MLP generate probability estimates that serve as inputs to a Logistic Regression meta-learner. Out-of-fold (OOF) predictions are used during development to ensure that the meta-learner is trained on predictions for observations excluded from the corresponding base-model training folds. This approach reduces the risk of information leakage and provides a more appropriate basis for learning model-combination weights.

However, the inclusion of additional learners does not guarantee an improvement over the strongest individual model. The project therefore compares the stacking framework with its base models using the same locked test set and complementary evaluation metrics.

## Probability Calibration and Probability of Default

An important distinction in credit-risk modelling is the difference between discrimination and probability calibration. A model may rank borrowers effectively according to relative default risk without producing predicted probabilities that closely match observed default frequencies.

Silva Filho et al. [16] review approaches for assessing and improving classifier calibration, providing a methodological foundation for evaluating probability reliability. This distinction is particularly relevant when the intended model output is interpreted as a Probability of Default (PD), rather than solely as a classification score.

The project's evaluation framework therefore incorporates several complementary metrics. ROC-AUC measures discrimination across classification thresholds, while PR-AUC provides additional information about performance on the less frequent default class. The Brier Score evaluates the accuracy of predicted probabilities by measuring their squared differences from observed outcomes.

Although F1 Score is reported, it is not the primary criterion for selecting the final PD output because performance at a fixed classification threshold, such as 0.5, does not directly establish probability reliability.

During model development, Platt scaling and isotonic calibration were considered as approaches to improving probability estimates. Calibration decisions were made using development-stage evidence before the final locked test evaluation. The final isotonic-calibrated output improved the Brier Score relative to the raw stacked predictions, although it did not outperform every individual model on all evaluation metrics.

The resulting PD estimates are connected to illustrative financial-risk calculations using:

**Expected Loss (EL) = PD × Loss Given Default (LGD) × Exposure at Default (EAD)**

This relationship demonstrates why probability quality is relevant to downstream financial interpretation. However, the resulting Expected Loss and pricing outputs are illustrative decision-support calculations rather than validated estimates for production lending decisions.

## Explainability, Financial Use and Research Gap

Explainability has become an important consideration in applying machine learning to financial decision-making. Bussmann et al. [17] investigate explainable machine learning in credit-risk management, while Černevičienė and Kabašinskas [18] review explainable artificial intelligence in finance. Talaat et al. [18] explore the integration of explainability and deep learning for credit-card default prediction, and Bahlool et al. [19] examine performance, fairness, and explainability in AI-based credit scoring.

Together, these studies highlight that predictive performance must be considered alongside the transparency and responsible application of credit-risk models.

The present project incorporates local SHAP explanations for the XGBoost component to help illustrate how selected borrower features contribute to individual model predictions. These explanations are not treated as causal relationships or as comprehensive explanations of the final calibrated stacking ensemble.

Similarly, the project's PD estimates, Expected Loss calculations, and indicative pricing scenarios are intended to demonstrate potential decision-support applications. They are not presented as automatic lending recommendations because real-world credit decisions require additional assessment of regulatory requirements, affordability, fairness, and institution-specific risk policies.

### Research Gap

Existing literature has examined credit-scoring algorithms, feature engineering, ensemble learning, probability calibration, and explainable artificial intelligence. However, differences in datasets, preprocessing approaches, validation procedures, and evaluation objectives make it difficult to compare findings directly across studies.

The research gap addressed by this project is therefore primarily integrative and methodological rather than algorithmic. The project does not seek to develop a new credit-scoring algorithm. Instead, it evaluates established modelling approaches using a common dataset, consistent data partitions, and comparable evaluation metrics, while allowing model-specific preprocessing and feature configurations.

Using the Home Credit Default Risk dataset, the framework integrates Logistic Regression, XGBoost, and MLP benchmarking with validation-driven feature engineering, five-fold out-of-fold stacking, and probability calibration. The resulting Probability of Default estimates are connected to illustrative Expected Loss calculations and indicative pricing scenarios, while local SHAP explanations provide insight into the XGBoost component.

The project further demonstrates how these components can be integrated into a locally deployable decision-support prototype using Flask, React, and Docker. Its main contribution is a reproducible end-to-end workflow that connects controlled model experimentation with probability-oriented evaluation, financial interpretation, and practical software implementation.

Nevertheless, the framework is evaluated using a single historical dataset, and its financial calculations and explanations are intentionally limited in scope. Consequently, the findings should be interpreted as evidence of the framework's feasibility within the selected experimental setting rather than proof of generalisability or readiness for production lending decisions.

## References

[1] X. Dastile, T. Çelik, and M. Potsane, “Statistical and machine learning models in credit scoring: A systematic literature survey,” *Applied Soft Computing*, vol. 91, Art. no. 106263, 2020, doi: 10.1016/j.asoc.2020.106263.

[2] S. Shi, R. Tse, W. Luo, S. D’Addona, and G. Pau, “Machine learning-driven credit risk: A systemic review,” *Neural Computing and Applications*, vol. 34, pp. 14327–14339, 2022, doi: 10.1007/s00521-022-07472-2.

[3] J. P. Noriega, L. A. Rivera, and J. A. Herrera, “Machine learning for credit risk prediction: A systematic literature review,” *Data*, vol. 8, no. 11, Art. no. 169, 2023, doi: 10.3390/data8110169.

[4] X. Zhang and L. Yu, “Consumer credit risk assessment: A review from the state-of-the-art classification algorithms, data traits, and learning methods,” *Expert Systems with Applications*, vol. 237, Art. no. 121484, 2024, doi: 10.1016/j.eswa.2023.121484.

[5] A. A. Montevechi, R. de Carvalho Miranda, A. L. Medeiros, and J. A. B. Montevechi, “Advancing credit risk modelling with machine learning: A comprehensive review of the state-of-the-art,” *Engineering Applications of Artificial Intelligence*, vol. 137, Art. no. 109082, 2024, doi: 10.1016/j.engappai.2024.109082.

[6] B. R. Gunnarsson, S. vanden Broucke, B. Baesens, M. Óskarsdóttir, and W. Lemahieu, “Deep learning for credit scoring: Do or don’t?,” *European Journal of Operational Research*, vol. 295, no. 1, pp. 292–305, 2021, doi: 10.1016/j.ejor.2021.03.006.

[7] Y. Liu, M. Yang, Y. Wang, Y. Li, T. Xiong, and A. Li, “Applying machine learning algorithms to predict default probability in the online credit market: Evidence from China,” *International Review of Financial Analysis*, vol. 79, Art. no. 101971, 2022, doi: 10.1016/j.irfa.2021.101971.

[8] A. Alonso Robisco and J. M. Carbó Martínez, “Measuring the model risk-adjusted performance of machine learning algorithms in credit default prediction,” *Financial Innovation*, vol. 8, Art. no. 70, 2022, doi: 10.1186/s40854-022-00366-1.

[9] S. R. Lenka, S. K. Bisoy, R. Priyadarshini, and M. Sain, “Empirical analysis of ensemble learning for imbalanced credit scoring datasets: A systematic review,” *Wireless Communications and Mobile Computing*, vol. 2022, Art. no. 6584352, 2022, doi: 10.1155/2022/6584352.

[10] A. A. Hussin Adam Khatir and M. Bee, “Machine learning models and data-balancing techniques for credit scoring: What is the best combination?,” *Risks*, vol. 10, no. 9, Art. no. 169, 2022, doi: 10.3390/risks10090169.

[11] M. Z. Abedin, C. Guotai, P. Hajek, and T. Zhang, “Combining weighted SMOTE with ensemble learning for the class-imbalanced prediction of small business credit risk,” *Complex & Intelligent Systems*, vol. 9, pp. 3559–3579, 2023, doi: 10.1007/s40747-021-00614-4.

[12] D. Yang, B. Xiao, M. Cao, and H. Shen, “A new hybrid credit scoring ensemble model with feature enhancement and soft voting weight optimization,” *Expert Systems with Applications*, vol. 238, Art. no. 122101, 2024, doi: 10.1016/j.eswa.2023.122101.

[13] R. Hlongwane, K. K. K. M. Ramaboa, and W. Mongwe, “Enhancing credit scoring accuracy with a comprehensive evaluation of alternative data,” *PLOS ONE*, vol. 19, no. 5, Art. no. e0303566, 2024, doi: 10.1371/journal.pone.0303566.

[14] Y. Zou, C. Gao, M. Xia, and C. Pang, “Credit scoring based on a bagging-cascading boosted decision tree,” *Intelligent Data Analysis*, vol. 26, no. 6, pp. 1557–1578, 2022, doi: 10.3233/IDA-216228.

[15] W. Yin, B. Kirkulak-Uludag, D. Zhu, and Z. Zhou, “Stacking ensemble method for personal credit risk assessment in peer-to-peer lending,” *Applied Soft Computing*, vol. 142, Art. no. 110302, 2023, doi: 10.1016/j.asoc.2023.110302.

[16] T. Silva Filho, H. Song, M. Perello-Nieto, R. Santos-Rodriguez, M. Kull, and P. Flach, “Classifier calibration: A survey on how to assess and improve predicted class probabilities,” *Machine Learning*, vol. 112, pp. 3211–3260, 2023, doi: 10.1007/s10994-023-06336-7.

[17] N. Bussmann, P. Giudici, D. Marinelli, and J. Papenbrock, “Explainable machine learning in credit risk management,” *Computational Economics*, vol. 57, pp. 203–216, 2021, doi: 10.1007/s10614-020-10042-0.

[18] J. Černevičienė and A. Kabašinskas, “Explainable artificial intelligence (XAI) in finance: A systematic literature review,” *Artificial Intelligence Review*, vol. 57, Art. no. 216, 2024, doi: 10.1007/s10462-024-10854-8.

[19] F. M. Talaat, A. Aljadani, M. Badawy, and M. Elhosseini, “Toward interpretable credit scoring: Integrating explainable artificial intelligence with deep learning for credit card default prediction,” *Neural Computing and Applications*, vol. 36, pp. 4847–4865, 2024, doi: 10.1007/s00521-023-09232-2.

[20] R. Bahlool, N. Hewahi, and W. Elmedany, “Performance, fairness, and explainability in AI-based credit scoring: A systematic literature review,” *Journal of Risk and Financial Management*, vol. 19, no. 2, Art. no. 104, 2026, doi: 10.3390/jrfm19020104.
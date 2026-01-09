# Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

![Book Cover](https://learning.oreilly.com/library/cover/9781492032632/250w/)

## 1. Repository Overview
**Purpose**: This repository serves as a comprehensive guide and practical implementation of the concepts presented in the book **"Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron (2nd Edition).

**Reference Book**: [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

**Learning Goals**:
- To bridge the gap between theoretical machine learning concepts and practical coding implementation.
- To master the use of production-ready Python frameworks: **Scikit-Learn**, **Keras**, and **TensorFlow**.
- To build an intuitive understanding of intelligent systems, from simple linear regressors to complex deep learning architectures.

---

## 2. Chapter-by-Chapter Summary

### Part I: The Fundamentals of Machine Learning
* **[Chapter 1: The Machine Learning Landscape](./Chapter_01_The_Machine_Learning_Landscape.ipynb)**
    * **Topics:** Supervised vs. Unsupervised, Batch vs. Online, Instance-based vs. Model-based learning.
    * **Key Challenge:** Overfitting, Underfitting, and Data Mismatch.
* **[Chapter 2: End-to-End Machine Learning Project](./Chapter_02_End_to_End_Machine_Learning_Project.ipynb)**
    * **Topics:** A full pipeline from fetching data to deployment.
    * **Skills:** Data cleaning (Pandas), Pipelines, ColumnTransformer, Cross-validation.
* **[Chapter 3: Classification](./Chapter_03_Classification.ipynb)**
    * **Topics:** Binary and Multiclass classification on MNIST.
    * **Metrics:** Precision, Recall, F1 Score, ROC Curves, Confusion Matrix.
* **[Chapter 4: Training Models](./Chapter_04_Training_Models.ipynb)**
    * **Topics:** Gradient Descent (Batch, Stochastic, Mini-batch), Polynomial Regression, Learning Curves.
    * **Regularization:** Ridge, Lasso, and Elastic Net.
* **[Chapter 5: Support Vector Machines](./Chapter_05_Support_Vector_Machines.ipynb)**
    * **Topics:** Linear SVC, Nonlinear SVM (Kernel Trick), SVR for regression.
    * **Concepts:** Large Margin Classification, Hard vs. Soft Margin.
* **[Chapter 6: Decision Trees](./Chapter_06_Decision_Trees.ipynb)**
    * **Topics:** Training, visualizing, and predicting with Trees (CART algorithm).
    * **Concepts:** Gini Impurity, Entropy, Regularization hyperparameters.
* **[Chapter 7: Ensemble Learning and Random Forests](./Chapter_07_Ensemble_Learning_and_Random_Forests.ipynb)**
    * **Topics:** Voting Classifiers, Bagging, Pasting, Random Forests.
    * **Boosting:** AdaBoost, Gradient Boosting, XGBoost.
* **[Chapter 8: Dimensionality Reduction](./Chapter_08_Dimensionality_Reduction.ipynb)**
    * **Topics:** The Curse of Dimensionality, Projection vs. Manifold Learning.
    * **Algorithms:** PCA, Kernel PCA, LLE (Locally Linear Embedding).
* **[Chapter 9: Unsupervised Learning Techniques](./Chapter_09_Unsupervised_Learning_Techniques.ipynb)**
    * **Topics:** Clustering (K-Means, DBSCAN) and Anomaly Detection (Gaussian Mixtures).
    * **Skills:** Elbow method, Silhouette score, Image segmentation.

### Part II: Neural Networks and Deep Learning
* **[Chapter 10: Introduction to ANN with Keras](./Chapter_10_Introduction_to_ANN_with_Keras.ipynb)**
    * **Topics:** Perceptrons, MLPs, Backpropagation.
    * **Skills:** Building models using Keras Sequential and Functional APIs.
* **[Chapter 11: Training Deep Neural Networks](./Chapter_11_Training_Deep_Neural_Networks.ipynb)**
    * **Topics:** Vanishing Gradients, Batch Normalization, Dropout.
    * **Optimization:** Momentum, RMSProp, Adam, Learning Rate Scheduling.
* **[Chapter 12: Custom Models and Training with TensorFlow](./Chapter_12_Custom_Models_and_Training_with_TensorFlow.ipynb)**
    * **Topics:** Low-level TensorFlow API (Tensors, Operations, GradientTape).
    * **Skills:** Custom Loss functions, Custom Layers, Custom Training Loops.
* **[Chapter 13: Loading and Preprocessing Data](./Chapter_13_Loading_and_Preprocessing_Data.ipynb)**
    * **Topics:** The `tf.data` API, TFRecords, Protocol Buffers.
    * **Skills:** Building high-performance ETL pipelines.
* **[Chapter 14: Deep Computer Vision Using CNNs](./Chapter_14_Deep_Computer_Vision_Using_CNNs.ipynb)**
    * **Topics:** Convolutional Layers, Pooling, ResNet, Xception.
    * **Skills:** Transfer Learning with pretrained models.
* **[Chapter 15: Processing Sequences Using RNNs and CNNs](./Chapter_15_Processing_Sequences_Using_RNNs_and_CNNs.ipynb)**
    * **Topics:** RNNs, LSTMs, GRUs, 1D CNNs (WaveNet).
    * **Skills:** Time series forecasting, sequence-to-vector models.
* **[Chapter 16: NLP with RNNs and Attention](./Chapter_16_NLP_with_RNNs_and_Attention.ipynb)**
    * **Topics:** Char-RNN, Encoder-Decoder, Beam Search, Attention Mechanisms.
    * **Architecture:** The Transformer (Self-Attention, Multi-Head Attention).
* **[Chapter 17: Autoencoders and GANs](./Chapter_17_Autoencoders_and_GANs.ipynb)**
    * **Topics:** Representation Learning, Variational Autoencoders (VAEs), GANs.
    * **Skills:** Dimensionality reduction, Generative modeling.
* **[Chapter 18: Reinforcement Learning](./Chapter_18_Reinforcement_Learning.ipynb)**
    * **Topics:** Markov Decision Processes (MDP), Q-Learning, DQN.
    * **Skills:** Building Agents for OpenAI Gym environments.
* **[Chapter 19: Training and Deploying at Scale](./Chapter_19_Training_and_Deploying_TF_Models_at_Scale.ipynb)**
    * **Topics:** TF Serving, TFLite (Mobile), Distributed Training.
    * **Skills:** Model export, Docker deployment, Multi-GPU training strategy.

### Appendices
* **[Appendix A: Machine Learning Project Checklist](./Appendix_A_ML_Project_Checklist.ipynb)**
    * A comprehensive 8-step checklist to guide you through any ML project, from framing the problem to system maintenance.

---

## 3. Tools & Technologies
* **Language**: Python 3.x
* **Core Libraries**:
    * **Scikit-Learn**: Traditional Machine Learning algorithms.
    * **TensorFlow 2.0 & Keras**: Deep Learning and Neural Networks.
    * **Pandas & NumPy**: Data manipulation and numerical analysis.
    * **Matplotlib & Seaborn**: Data visualization.
* **Environment**: Jupyter Notebook / Google Colab

## 4. Learning Outcome
By completing this repository, a student will gain a robust theoretical foundation in Machine Learning and Deep Learning, alongside the practical engineering skills required to build, train, debug, and deploy intelligent systems in real-world scenarios.

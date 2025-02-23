# Pratilipi Recommendation System

This repository contains a hybrid recommendation system that predicts which pratilipis (stories) a user is likely to read in the future. The model leverages both collaborative signals from user interactions and content-based signals from pratilipi metadata to generate personalized recommendations.

---

## Repository Structure

- **data_preprocessing.ipynb**  
  Notebook for data preprocessing and exploratory data analysis (EDA).

- **hybrid_recc_system.ipynb**  
  Notebook for model training, evaluation, and generating recommendations.

- **requirements.txt**  
  List of Python dependencies.

- **README.md**  
  This file, which includes setup instructions, documentation on data analysis and preprocessing, model choice, training iterations, and future work.

- **user_recommendations.csv**
  CSV file containing user recommendations of the test data.
  
---

## How to Run the Code

Follow these steps to set up and run the project:

1. **Fork the Repository**

   - Click the "Fork" button on the GitHub page to create your own copy.
2. **Clone the Repository**

   Open your terminal and run:
   ```bash
   git clone https://github.com/your-username/repository-name.git
3. **Create a Python Virtual Environment**

    Navigate to the repository directory and create a virtual environment:
    ```bash
    cd repository-name
    python3 -m venv pa_env
4. **Activate the Virtual Environment**

    Activate the virtual environment:
    ```bash
    source pa_env/bin/activate
5. **Install Dependencies**

    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
6. **Run the Jupyter Notebook for Data Preprocessing and EDA**

    Open the `data_preprocessing.ipynb` notebook in Jupyter Notebook and run the cells to preprocess the data and perform EDA.
7. **Run the python script for Model Training and Evaluation**

    ```bash
        python hybrid_recc_system.py
---

## Documentation

### Data Analysis and Preprocessing

#### Data Cleaning & Preparation:
- **Cleaning:** Duplicate records were removed and missing values handled.
- **Timestamp Conversion:** Timestamps were converted to datetime objects for proper handling.

#### Feature Engineering:
- Derived temporal features such as interaction hour and day.
- Computed user-level statistics (e.g., average reading percentage and reading time) to capture behavioral patterns.

#### Exploratory Data Analysis (EDA):
- **Distribution Analysis:**
  - Plotted histograms for reading percentages and line charts for user activity patterns.
- **Category Analysis:**
  - Created bar charts to visualize the distribution of pratilipi categories.
- **Relationship Exploration:**
  - Used scatter plots to explore correlations (e.g., reading time vs. read percentage).

The insights from EDA helped guide further feature engineering and provided a solid foundation for the model.

### Model Choice

#### Why a Hybrid Model?
- **Combining Strengths:**
  - The hybrid model integrates collaborative filtering (from user interaction data) with content-based filtering (from pratilipi metadata such as category), improving recommendation quality even in sparse data situations.
- **Cold-Start Handling:**
  - Including content features enables recommendations for new or less popular pratilipis that lack sufficient interaction history.

#### Why LightFM?
- **Hybrid Architecture:**
  - LightFM's architecture naturally supports both collaborative and content-based approaches, making it ideal for our hybrid recommendation needs.
- **Scalability:**
  - Efficient implementation that can handle large-scale datasets with millions of users and items.
- **Feature Integration:**
  - Built-in support for incorporating user and item features, allowing seamless integration of metadata.
- **Implicit Feedback:**
  - Specifically designed to work well with implicit feedback data (like reading interactions) through specialized loss functions like WARP.
- **Production Ready:**
  - Proven track record in production environments with good performance characteristics.

#### Metric Selection – AUC:
The AUC (Area Under the ROC Curve) metric was chosen to measure the model's ability to correctly rank positive instances higher than negative ones. This is crucial for implicit feedback data where ranking quality is more significant than predicting exact scores.

### Model Training and Iterative Improvements

The model training underwent several iterations to improve performance:

1. **Initial Model:**
   - Setup: Basic LightFM model using the WARP loss function.
   - Outcome: Established a baseline.
   - Evaluation: AUC score of 0.7103

2. **Increasing Epochs & Adjusting Learning Rate:**
   - Modification: Increased epochs to 50 and reduced the learning rate to 0.01.
   - Outcome: Slight improvements were observed as the model had more iterations to learn latent factors.
   - Evaluation: AUC score of 0.75

3. **Loss Function Experimentation:**
   - Attempt: Tested Bayesian Personalized Ranking (BPR) loss.
   - Outcome: Performance decreased, leading to a reversion to the original WARP loss.
   - Evaluation: AUC score of 0.68

4. **Stratified Sampling:**
   - Approach: Implemented stratified sampling based on users' interaction counts to preserve the overall distribution.
   - Outcome: Improved evaluation metrics by ensuring that users are consistently represented in both training and test sets.
   - Evaluation: AUC score of 0.8109

5. **Data Augmentation:**
   - Improvement: Augmented user data with temporal features (most common interaction hour and day) to provide additional context.
   - Outcome: Mitigated data sparsity by adding richer user features but unfortunately the model performance decreased significantly.
   - Evaluation: AUC score of 0.61

### Final Remarks and Future Work

While iterative improvements have led to better evaluation metrics, the overall performance is still limited by the simplicity of the available metadata. The current metadata includes only basic features like category and reading time. In real-world applications, richer metadata is essential for high-quality recommendations.

#### Future Enhancements:

1. **Enrich Metadata:**
   - A promising future direction is to apply multi-label classification or topic modeling on pratilipi content. This would generate more informative and detailed features, further boosting the system's recommendation quality.

2. **Additional Feature Engineering:**
   - Exploring other user and content signals, such as session-based behavior or more granular content descriptors, could also improve performance.


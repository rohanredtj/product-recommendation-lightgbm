# Product Recommendation Lightgbm

This repository contains an advanced product recommendation system that utilizes machine learning techniques to predict customer purchases and enhance sales strategies. The system integrates data preprocessing, feature engineering, and model optimization to improve customer targeting and increase sales efficiency.

## Project Overview

The Product Recommendation System leverages LightGBM's gradient boosting capabilities combined with Optuna's hyperparameter optimization to deliver accurate, data-driven recommendations for businesses. This allows companies to optimize their product offerings and sales approaches based on predicted customer behavior.

## Key Features

- Automated data preprocessing pipeline
- Advanced feature engineering for time series data
- LightGBM model with Optuna hyperparameter optimization
- Cross-validation for robust model evaluation
- Customized evaluation metrics for business relevance
- Client-level and visit-level precision analysis

## Technical Stack

- **Programming Language**: Python
- **Libraries**:
  - pandas: For data manipulation and analysis
  - scikit-learn: For machine learning utilities
  - LightGBM: For gradient boosting framework
  - Optuna: For hyperparameter optimization
  - NumPy: For numerical computing
- **Tools**:
  - Tmux: For terminal session management

## System Architecture

The system consists of four main components:

1. **Data Preprocessing** (`pipeline1_data_preprocessing.py`):
   - Loads and merges time series and attribute data
   - Performs data cleaning and normalization
   - Splits data into training and testing sets

2. **Model Training** (`pipeline2.2_model_training_with_optuna.py`):
   - Implements Optuna for hyperparameter tuning
   - Trains LightGBM model with optimized parameters
   - Utilizes k-fold cross-validation for robust evaluation

3. **Model Evaluation** (`pipeline3.2_model_evaluation_for_with_optuna_models.py`):
   - Computes various performance metrics (accuracy, precision, recall, AUC)
   - Generates detailed evaluation reports
   - Performs client-level and visit-level precision analysis

4. **Model Inference** (`pipeline4_model_inference.py`):
   - Applies trained model to generate predictions
   - Produces recommendation lists based on model outputs

## Key Functionalities

- **Time Series Feature Engineering**: Standardizes weekly sales data for improved model input
- **Hyperparameter Optimization**: Uses Optuna to find optimal LightGBM parameters
- **Custom Evaluation Metric**: Implements "overall precision after trimming" for business-relevant model selection
- **Recommendation Generation**: Ranks products and filters based on recommended quantity

## Performance Metrics

- Cross-validated performance metrics (accuracy, precision, recall, AUC)
- Client-level and visit-level precision analysis
- Custom evaluation metric for business-specific performance assessment

## Impact and Benefits

- Improved accuracy in predicting customer purchases
- Enhanced customer targeting for more effective sales strategies
- Automated recommendation system reducing manual effort in sales planning
- Data-driven insights for inventory management and product placement

## Challenges Addressed

- Handling time series data in a product recommendation context
- Balancing model complexity with interpretability for business use
- Incorporating business logic (recommended quantity) into the ML pipeline
- Developing custom evaluation metrics aligned with business goals

## Getting Started

To use this product recommendation system, follow these steps:

1. Clone this repository to your local machine:
```bash
git clone https://github.com/rohanredtj/product-recommendation-system.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the data preprocessing pipeline:
```bash
python pipeline1_data_preprocessing.py
```

4. Train the model with Optuna optimization:
```bash
python pipeline2.2_model_training_with_optuna.py
```

5. Evaluate the model performance:
```bash
python pipeline3.2_model_evaluation_for_with_optuna_models.py
```

6. Generate recommendations:
```bash
python pipeline4_model_inference.py
```

## Future Enhancements

- Integration with real-time data streams for dynamic recommendations
- Incorporation of additional data sources (e.g., customer demographics, market trends)
- Development of a user interface for easy interaction with the recommendation system
- Exploration of ensemble methods to further improve prediction accuracy

## Conclusion

The Product Recommendation System represents a significant advancement in applying machine learning to sales and inventory management. By combining LightGBM's powerful gradient boosting capabilities with Optuna's intelligent hyperparameter optimization, the system delivers highly accurate purchase predictions. This project not only enhances sales predictions but also provides a robust foundation for data-driven decision-making in sales strategy, inventory management, and customer relationship management.

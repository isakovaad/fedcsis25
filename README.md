# Chess Puzzle Difficulty Prediction
## FedCSIS 2025 Challenge

A machine learning project to predict chess puzzle difficulty ratings using chess position analysis and player success probability data.

## Competition Overview

This project was developed for the **FedCSIS 2025 Challenge**, where the goal is to predict how difficult a chess puzzle is based on:
- Initial chess position (FEN notation)
- Solution moves sequence
- Player success probability data across different rating levels
- Puzzle metadata and themes

Puzzle difficulty is measured using **Glicko-2 ratings** calibrated by Lichess, where each puzzle attempt is treated as a match between the user and the puzzle.

## Dataset

### Training Data
- **4.56M chess puzzles** with complete feature sets
- **Rating range**: 399 to 3,284 (mean: ~1,501, std: ~544)
- **Features**: 32 columns including success probabilities, themes, and metadata

### Test Data
- **2,235 puzzles** for final predictions
- **25 features** (subset of training columns)

### Key Features
- **Success Probabilities**: Player success rates across different rating levels (1050-2050) for both rapid and blitz games
- **FEN Strings**: Chess position notation
- **Move Sequences**: Solution moves in algebraic notation
- **Themes**: Puzzle categories (tactics, endgames, checkmates, etc.)
- **Metadata**: Play count, popularity, rating deviation

## Project Structure

```
├── data_exploration.ipynb           # Initial data analysis and baseline model
├── feature_engineering&models.ipynb # Advanced feature engineering and models
├── training_data_02_01.csv         # Original training dataset
├── testing_data_cropped.csv        # Original test dataset
├── train_cleaned.csv               # Preprocessed training data
├── test_cleaned.csv                # Preprocessed test data
├── best_hybrid_model.pth           # Saved neural network weights
├── corrected_submission.txt        # Final predictions
└── enhanced_submission.txt         # Alternative predictions
```

## Technical Approach

### 1. Data Preprocessing
- **Missing Value Handling**: Strategic imputation for numerical and categorical features
- **Feature Cleaning**: Standardized text processing and numerical normalization

### 2. Feature Engineering
- **Success Probability Analysis**: Statistical features from player performance data
  - Mean, std, min/max across rating levels
  - Rapid vs Blitz performance differences
  - Difficulty slope calculations
- **Chess Position Features**: FEN string parsing using `python-chess`
  - Material balance and piece counts
  - King positions and game phase detection
  - Legal moves and check status
- **Move Sequence Analysis**: Solution complexity metrics
  - Capture and check frequencies
  - Move type distribution
  - Sequence length and complexity
- **Theme Features**: Categorical puzzle type indicators
- **Interaction Features**: Cross-feature relationships

### 3. Model Development

#### Baseline Model (XGBoost)
- **RMSE**: 337 on validation set
- **Features**: 36 basic features
- **Training Time**: <1 minute on 100K samples

#### Advanced Models
1. **Hybrid Tree+Neural Model**
   - LightGBM for structured features
   - Neural network for complex interactions
   - **RMSE**: 544.4 (underperformed baseline)

2. **Transformer Architecture** (Experimental)
   - Multi-head attention for feature relationships
   - Failed due to tensor dimension mismatches

## Results

**Note: This project is currently in development. The results below represent initial experiments, and the files will be updated soon with improved models and final competition results.**

| Model | RMSE | Improvement vs Baseline |
|-------|------|------------------------|
| XGBoost Baseline | 337.0 | - |
| Hybrid Model | 544.4 | -61.5% |
| Transformer | Failed | - |

**Key Insight**: Simple tree-based models outperformed complex neural architectures for this tabular data problem.

## Feature Importance Analysis

Top performing features (from baseline model):
1. `success_prob_rapid_1150` (59.2% importance)
2. `success_prob_rapid_1050` (13.3% importance)
3. `success_prob_blitz_1050` (11.0% importance)
4. `move_count` (3.6% importance)
5. `rapid_min` (1.9% importance)

**Finding**: Success probability features dominated model performance, indicating that historical player performance is the strongest predictor of puzzle difficulty.

## Dependencies

```python
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.1.0
xgboost>=1.6.0
lightgbm>=3.3.0
torch>=1.12.0
python-chess>=1.999
matplotlib>=3.5.0
```

## Usage

### Quick Start
```python
# Load and run baseline model
import pandas as pd
from data_exploration import ultra_fast_xgboost_baseline

# Load data
train_df = pd.read_csv('train_cleaned.csv')
test_df = pd.read_csv('test_cleaned.csv')

# Train baseline model
model, feature_importance = ultra_fast_xgboost_baseline(train_features, train_df)

# Generate predictions
predictions = model.predict(test_features)
```

### Advanced Feature Engineering
```python
from feature_engineering_models import run_feature_engineering

# Create advanced features
train_features, test_features = run_feature_engineering(
    train_df, test_df, sample_size=100000
)
```

## Model Performance Analysis

### Validation Metrics
- **RMSE as % of std dev**: 61.9% (reasonable given rating variance)
- **Prediction range**: Covers full rating spectrum (400-3300)
- **Cross-validation**: Consistent performance across folds

### Key Challenges
1. **Class Imbalance**: Ratings concentrated around 1500
2. **Feature Noise**: Many chess-specific features didn't improve performance
3. **Overfitting**: Complex models struggled with generalization

## Future Improvements

1. **Data Augmentation**: Generate synthetic puzzles with known difficulties
2. **Ensemble Methods**: Combine multiple model architectures
3. **Chess Engine Integration**: Incorporate position evaluation scores
4. **Time-Series Features**: Account for rating evolution over time
5. **Transfer Learning**: Pre-train on chess position databases

## Competition Results

- **Final Submission**: `corrected_submission.txt`
- **Model Used**: XGBoost baseline (best performer)
- **Feature Count**: 36 engineered features
- **Training Size**: 100K samples for efficiency

## Contributing

This project was developed for the FedCSIS 2025 Challenge. Feel free to:
- Experiment with different feature engineering approaches
- Try alternative model architectures
- Improve chess position analysis methods

## References

1. [FedCSIS 2025 Challenge](https://fedcsis.org/2025/challenge)
2. [Lichess Puzzle Database](https://database.lichess.org/)
3. [Glicko-2 Rating System](http://www.glicko.net/glicko.html)
4. [Python Chess Library](https://python-chess.readthedocs.io/)

## License

This project is open source and available under the MIT License.

---

*"In chess, as in machine learning, the position tells the story, but the patterns reveal the truth."*

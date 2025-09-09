# Volleyball Group Activity Recognition

This repository implements a comprehensive framework for **Group Activity Recognition (GAR)** in volleyball videos, based on the research paper:

**[A Hierarchical Deep Temporal Model for Group Activity Recognition](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)**  
*Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, Greg Mori. IEEE Computer Vision and Pattern Recognition 2016*

## 🏐 Overview

This project focuses on recognizing complex group activities in volleyball games by analyzing the temporal dynamics and spatial relationships between multiple players. The system can identify various volleyball-specific group activities such as right/left team spikes, sets, passes, and winning points.

## 🎯 Problem Statement

Group Activity Recognition in sports videos is challenging because it requires:
- Understanding individual player actions
- Modeling temporal dependencies across frames
- Capturing spatial relationships between multiple players
- Recognizing coordinated team activities

## 🏗️ Architecture

The implementation includes **8 different baseline models** that progressively increase in complexity:

### Model Variants

| Model | Description | Architecture | Test Accuracy |
|-------|-------------|--------------|---------------|
| **Baseline 1** | Simple ResNet-50 | Single CNN for frame-level classification | **74.83%** |
| **Baseline 3A** | Feature extraction model | ResNet-based feature extractor for individual actions | **78.27%** |
| **Baseline 3B** | Enhanced feature model | Improved version of 3A with better feature representation | **82.12%** |
| **Baseline 4** | Temporal modeling | LSTM-based temporal sequence modeling | **81.08%** |
| **Baseline 5** | Multi-stream approach | Multiple input streams for different modalities | **83.40%** |
| **Baseline 6** | Attention mechanism | Attention-based temporal modeling | **77.86%** |
| **Baseline 7** | Hierarchical modeling | Multi-level temporal and spatial modeling | **86.46%** |
| **Baseline 8** | Advanced LSTM | Dual LSTM with team-based aggregation | **89.08%** |

### Key Components

- **Feature Extraction**: Uses pre-trained ResNet models to extract 2048-dimensional features from player bounding boxes
- **Temporal Modeling**: LSTM networks to capture sequential dependencies
- **Spatial Aggregation**: Team-based feature aggregation for group activity recognition
- **Multi-level Classification**: Hierarchical approach from individual actions to group activities

## 📊 Dataset & Annotations

The system works with volleyball video datasets containing:
- **Individual Actions**: 9 action classes (waiting, setting, digging, falling, spiking, blocking, jumping, moving, standing)
- **Group Activities**: 8 group activity classes (r_spike, r_set, r-pass, r_winpoint, l-spike, l_set, l-pass, l_winpoint)
- **Player Tracking**: Bounding box annotations for each player across video frames
- **Temporal Segmentation**: Video clips annotated with group activity labels

## 🚀 Getting Started

### Prerequisites

- Python 3.13.7+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CUDA (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd volleyball
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies using uv:
```bash
uv sync
```

This will automatically:
- Create a virtual environment
- Install all dependencies from `pyproject.toml`
- Set up the project for development

### Project Configuration

The project uses modern Python packaging with `pyproject.toml`:

- **Dependencies**: PyTorch, torchvision, Pillow, matplotlib, scikit-learn
- **Python Version**: 3.13.7+
- **Code Quality**: Ruff for linting and formatting
- **Package Manager**: uv for fast dependency resolution

### Project Structure

```
volleyball/
├── models/                 # Model architectures (8 baselines)
├── datasets/              # Dataset loaders and utilities
├── trainers/              # Training scripts for each baseline
├── utils/                 # Utility functions and helpers
├── extract_features.py    # Feature extraction pipeline
├── constants.py           # Configuration constants
├── trained_models/        # Pre-trained model weights
├── confusion_matrix/      # Confusion matrix visualizations
├── loss_accuracy/         # Training curves
└── logs/                  # Training logs
```

## 🔧 Usage

### 1. Feature Extraction

Extract deep features from video frames using pre-trained models:

```bash
uv run python extract_features.py
```

This script:
- Loads video frames and player bounding boxes
- Crops individual player regions
- Extracts 2048-dimensional features using Baseline 3A model
- Saves features for training/validation/test splits

### 2. Training Models

Train any of the 8 baseline models:

```bash
# Train Baseline 1 (ResNet-50)
uv run python trainers/train_b1.py

# Train Baseline 8 (Advanced LSTM)
uv run python trainers/train_b8.py
```

### 3. Model Evaluation

Models are automatically evaluated on test sets during training, generating:
- Confusion matrices for training and validation
- Loss and accuracy curves
- F1 scores and classification metrics

## 📈 Results & Performance

### Performance Summary

The repository includes comprehensive evaluation results with **Baseline 8** achieving the highest test accuracy of **89.08%**:

- **Best Performing Model**: Baseline 8 (Advanced LSTM) - 89.08% accuracy
- **Strong Temporal Models**: Baseline 7 (86.46%) and Baseline 5 (83.40%) show the importance of temporal modeling
- **Feature Quality Impact**: Baseline 3B (82.12%) outperforms 3A (78.27%), demonstrating improved feature representation
- **Baseline Comparison**: Simple ResNet-50 (Baseline 1) achieves 74.83%, providing a solid foundation

### Key Insights

1. **Temporal Modeling is Critical**: Models with LSTM components (B4, B5, B7, B8) consistently outperform frame-based approaches
2. **Team-based Aggregation Works**: Baseline 8's dual LSTM with team-based feature aggregation achieves the best results
3. **Hierarchical Approaches Excel**: Multi-level modeling (B7, B8) captures both individual and group dynamics effectively
4. **Feature Quality Matters**: Enhanced feature extraction (B3B vs B3A) provides measurable improvements

### Available Results

- **Training Curves**: Loss and accuracy plots for all baselines
- **Confusion Matrices**: Detailed classification performance analysis
- **Model Weights**: Pre-trained models for immediate use
- **Logs**: Detailed training logs for reproducibility

## 🔬 Research Contributions

This implementation provides:

1. **Comprehensive Baselines**: 8 different approaches to GAR
2. **Volleyball-Specific Modeling**: Domain-adapted for sports analysis
3. **Temporal Dynamics**: Advanced LSTM-based sequence modeling
4. **Spatial Relationships**: Team-based feature aggregation
5. **Reproducible Results**: Complete training and evaluation pipeline

## 🛠️ Development

### Development Workflow

```bash
# Activate the virtual environment
uv shell

# Run scripts in the virtual environment
uv run python script.py

# Add new dependencies
uv add package-name

# Update dependencies
uv sync

# Run linting and formatting
uv run ruff check .
uv run ruff format .
```

### Customization

### Adding New Models

1. Create a new model class in `models/`
2. Implement the forward pass
3. Create a training script in `trainers/`
4. Update constants if needed

### Modifying Activities

Edit `constants.py` to:
- Add new individual actions
- Define new group activities
- Adjust feature dimensions

## 📚 References

- [Original Paper](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)
- [CVPR 2016](https://cvpr2016.thecvf.com/)
- [Group Activity Recognition Survey](https://arxiv.org/abs/2006.06966)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new model architectures
- Improve documentation
- Add new evaluation metrics

## 📄 License

This project is for research purposes. Please cite the original paper if you use this implementation in your research.

## 🙏 Acknowledgments

- Original authors for the foundational research
- PyTorch community for the deep learning framework
- Computer vision research community for open-source tools and datasets

---

**Note**: This implementation focuses on volleyball group activity recognition and serves as a comprehensive baseline for sports video analysis research.
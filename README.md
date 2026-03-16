# Food Image Classification — Custom CNN (PyTorch)

A custom convolutional neural network built from scratch for fine-grained food image classification across **80 categories** on a dataset of ~27,000 images.

Built as part of the [Food Recognition Challenge 2026](https://www.kaggle.com/) on Kaggle.

## Results
| Metric | Score |
|--------|-------|
| Test Accuracy | **63%** |
| Classes | 80 food categories |
| Dataset size | ~27,000 images |

> Baseline (random): ~1.25% — model achieves ~50× improvement over random.

## Architecture
- Custom **ResNet-style blocks** with skip connections
- Kaiming normal weight initialisation
- Optional BatchNorm per block
- Global average pooling → fully connected classifier

## Training Details
- **Optimizer:** Adam with warm-up + cosine annealing LR schedule (`SequentialLR`)
- **Augmentation:** RandomResizedCrop, RandomHorizontalFlip, normalisation (ImageNet stats)
- **Class imbalance:** handled via `WeightedRandomSampler`
- **Framework:** PyTorch

## Stack
`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `NumPy` · `pandas`

## File Structure
```
food-image-classification/
├── food-class-v2.ipynb   # Main notebook: model, training, evaluation
└── README.md
```

## How to Run
1. Download the dataset from the competition page
2. Open `food-class-v2.ipynb` in Kaggle or Jupyter
3. Update the dataset paths in Section 2
4. Run all cells

## Author
**Jan Kawak** — AI Student @ University of Amsterdam  
[LinkedIn](https://linkedin.com/in/Jan-Kawak) · [GitHub](https://github.com/JanKawak)

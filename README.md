# Food Image Classification — Custom CNN with Ensemble (PyTorch)

A custom convolutional neural network built from scratch for fine-grained food image classification across **80 categories** on ~27,000 images. Built for the [Food Recognition Challenge 2026](https://www.kaggle.com/competitions/food-recognition-challenge-2026) on Kaggle.

## Results
| Configuration | Test Accuracy |
|---------------|---------------|
| Single model (best) | 63% |
| Ensemble (3 models, 20 TTA) | **best submission** |

> Baseline (random): ~1.25% — model achieves ~50× improvement over random.

## Architecture

Custom **ResNet-style CNN** built from scratch with:
- Residual blocks (`ResBlock`) with skip connections
- Optional **Squeeze-and-Excitation (SE)** attention
- Kaiming normal weight initialisation
- Optional BatchNorm per block
- Global average pooling → fully connected classifier

```
Input (3×224×224)
    → ResBlock(3→64)
    → ResBlock(64→128)
    → ResBlock(128→256)
    → ResBlock(256→512)
    → ResBlock(512→512)
    → GlobalAvgPool
    → Linear(512→80)
```

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Architecture | `[64, 128, 256, 512, 512]` |
| Batch size | 128 |
| Epochs | 77 |
| Learning rate | 1e-3 |
| LR schedule | Linear warmup → Cosine annealing |
| Weight decay | 5e-4 |
| BatchNorm | ✅ |
| SE attention | ✅ |
| Val fraction | 10% |

**Data augmentation:**
- RandomResizedCrop(224, scale=0.7–1.0)
- RandomHorizontalFlip, RandomVerticalFlip
- RandomRotation(10°)
- ColorJitter (brightness, contrast, saturation, hue)
- RandomGrayscale
- RandomErasing
- ImageNet normalisation

**Class imbalance:** handled via `WeightedRandomSampler`

## Ensemble & Test-Time Augmentation (TTA)

Final submission uses an ensemble of 3 models trained with different seeds, with **20 TTA passes** per model:

| Model | Seed | SE |
|-------|------|----|
| model1 | 42  | ❌ |
| model2 | 123 | ❌ |
| model3 | 456 | ✅ |

Logits are accumulated across all models and TTA passes before taking argmax.

> Pre-trained weights are not included in this repo.  
> Download from [Kaggle Models — janjunior/food-models](https://www.kaggle.com/models/janjunior/food-models)  
> Or reproduce by running the notebook from top to bottom.

## Notebook Structure

| Section | Description |
|---------|-------------|
| 1. Imports & Setup | Libraries, device, seeds |
| 2. Dataset & Data Loading | `FoodDataset`, transforms, dataloaders |
| 3. CNN Architecture | `ResBlock`, `CNN` class |
| 4. Training Utilities | Accuracy, evaluation, submission helpers |
| 5. Training Loop | Full training with scheduler, early stopping |
| 6. Experiment Config | Hyperparameters and paths |
| 7. Ensemble & Submission | 3-model ensemble with 20 TTA passes |

## Stack
`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `NumPy` · `pandas` · `Kaggle`

## How to Run
1. Open on Kaggle and add the competition dataset
2. (Optional) Add pre-trained weights from [janjunior/food-models](https://www.kaggle.com/models/janjunior/food-models)
3. Run all cells — trains 3 models and generates `submission.csv`

## Author
**Jan Kawak** — AI Student @ University of Amsterdam  
[LinkedIn](https://www.linkedin.com/in/jankawak) · [GitHub](https://github.com/JanKawak)

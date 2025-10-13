# 🐟 MaskAugFish
**Exploring the effects of mask-aware data augmentation on CNN-based fish species recognition**

---

## 📘 Overview
Fish species recognition is a challenging task in marine computer vision due to the strong reliance on subtle color patterns, shapes, and environmental variations.  
This project investigates **how different data augmentation strategies affect model performance** in fish identification tasks — specifically when **color and texture are key discriminative features**.

We compare four augmentation regimes:
1. **No Augmentation** — baseline model trained on raw images.  
2. **Mask-Only Augmentation** — transformations applied only to the fish region.  
3. **Background-Only Augmentation** — transformations applied only to the background while preserving fish color/patterns.  
4. **Whole-Image Augmentation** — standard full-frame augmentations.

By isolating augmentation scopes, we aim to understand how augmentations influence a CNN’s ability to learn the correct visual cues for species classification.

---

## 🎯 Research Objective
> **To evaluate how mask-aware vs. global vs. no augmentation strategies affect fish species recognition performance when color and morphology are the dominant features.**

Key questions:
- Does augmenting the entire image distort color-based features critical to fish ID?  
- Can background-only augmentation improve robustness to environmental changes?  
- Which augmentation scope best balances generalization and feature integrity?

---

## 🐠 Dataset: Fish4Knowledge
We use the **[Fish4Knowledge](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/) dataset**, which provides:
- Labeled images of tropical reef fish across multiple species.  
- Pixel-level **fish masks** for segmentation-based processing.  
- Naturally **imbalanced class distribution**, reflecting real-world marine diversity.  

---

## 🧠 Methodology
### Model
- **Base architecture:** ResNet-50 CNN (pretrained on ImageNet).  
- **Input size:** 224 × 224.  
- **Loss function:** Cross-entropy with class-weighting to handle imbalance.  
- **Optimizer:** ADAM (Tentative) 

### Augmentation Pipeline
| Regime | Augmentation Target | Example Transforms |
|:-------:|:--------------------|:-------------------|
| No Aug | None | - |
| Mask-Only | Fish region | Flip, rotation, mild brightness/contrast |
| Background-Only | Background pixels only | Blur, color cast, noise, hue shift |
| Whole-Image | Entire frame | Standard random crop, color jitter, rotation |

Mask regions are extracted using Fish4Knowledge segmentation masks. 

---

## 📊 Evaluation
- **Metrics:** Accuracy, Macro-F1, Balanced Accuracy.  
- **Visualization:** Confusion matrix, and t-SNE embeddings.  
- **Analysis:**  
  - Compare species-specific confusion (especially color-similar fish).  
  - Test background leakage effects.  
  - Evaluate model focus via Grad-CAM on fish vs. background regions.

---

## 📈 Expected Outcomes
- Whole-image and/or Mask-Only augmentation may reduce performance due to distortion of discriminative color cues.  
- Background-only augmentation is expected to improve robustness to lighting and environment changes without harming feature learning.  


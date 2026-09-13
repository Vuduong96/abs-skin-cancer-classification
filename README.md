# Adaptive Balanced Sampling (ABS) Active Learning Strategy for Skin Cancer Classification

Most active learning methods rank unlabeled samples using uncertainty scores computed globally across all classes. Under class imbalance, this often causes acquisition to be dominated by majority classes, leaving clinically important minority categories under-represented even as the labeling budget increases.

ABS addresses this limitation through a class-aware acquisition strategy:

### Phase 1 — Balanced Candidate Allocation
The unlabeled pool is partitioned according to the model's predicted classes. A rarity-scaled quota is then assigned to each class, ensuring that under-represented categories receive sufficient representation within the candidate pool.

### Phase 2 — Informative Sample Selection
Samples are selected from the balanced candidate pool by jointly considering predictive uncertainty and feature-space diversity, reducing redundancy while retaining informative examples.

By combining class-balanced allocation with uncertainty- and diversity-based selection, ABS improves minority-class performance while maintaining strong performance on majority classes. The framework is evaluated on three dermatological benchmarks (HAM10000, PAD-UFES-20, and ISIC-2020) using patient- and lesion-level group-aware splits to better assess generalization to previously unseen patients and lesions.

## ✨ Highlights

- **Class-aware active learning for imbalanced medical datasets**
  - Explicitly incorporates class balance into the acquisition process rather than relying solely on global uncertainty.

- **Two-phase acquisition strategy**
  - Phase 1: rarity-aware, class-balanced candidate allocation.
  - Phase 2: uncertainty-diversity selection from the balanced candidate pool.

- **Improved minority-class performance**
  - Designed to increase representation of clinically important but under-represented lesion categories during active learning.

- **Evaluated on multiple dermatology benchmarks**
  - Experiments conducted on HAM10000, PAD-UFES-20, and ISIC-2020 under group-aware evaluation protocols.

- **Strong performance under severe class imbalance**
  - Improves melanoma F1 from **0.50 → 0.80** on PAD-UFES-20 at a matched labeling budget.
  - Improves dermatofibroma F1 from **0.55 → 0.67** on HAM10000.

- **Reliable evaluation protocol**
  - Uses patient-level and lesion-level group-aware splitting to reduce overlap between training and evaluation samples.

- **Annotation-efficient learning**
  - Achieves competitive performance while using substantially fewer labeled samples than fully supervised training.

- **Complementary acquisition phases**
  - Ablation and statistical analyses show that both the class-balanced allocation stage and the uncertainty-diversity selection stage contribute to performance improvements.

- **Insights beyond class imbalance**
  - Demonstrates that some classes remain challenging due to intrinsic visual ambiguity, highlighting the complementary roles of balanced acquisition and representation learning.

---

## 🧪 Repository Structure

```text
├── common/
├── pad20/
├── ham10000/
├── isic2020/
├── data/
└── outputs/
```

---

## 📊 Datasets

### PAD-UFES-20

- 2,298 smartphone-acquired clinical images
- 6 diagnostic categories
- Patient-level grouped split

### HAM10000

- 10,015 dermoscopic images
- 7 diagnostic categories
- Lesion-level grouped split

### ISIC-2020

- 33,126 dermoscopic images
- Binary task (malignant vs. benign)
- Patient-level grouped split

Please see `data/README.md` for dataset preparation instructions.

---

## ⚙️ Setup

```bash
git clone https://github.com/<anonymized>/abs-active-learning.git
cd abs-active-learning
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python pad20/train_full_abs_multiclass.py
python pad20/train_full_abs_binary.py
python ham10000/train_full_abs_multiclass.py
python ham10000/train_full_abs_binary.py
```

---

## 📈 Reported Results

| Dataset | Task | Accuracy | Macro-F1 | Balanced Accuracy |
|----------|----------|----------|----------|----------|
| PAD-UFES-20 | Multiclass | 0.755 | 0.718 | 0.688 |
| PAD-UFES-20 | Binary | 0.862 | 0.862 | 0.862 |
| HAM10000 | Multiclass | 0.858 | 0.772 | 0.729 |
| HAM10000 | Binary | 0.888 | 0.776 | 0.750 |

---

## 📖 Citation

```bibtex
@article{anonymous2026abs,
  title={[Title redacted for double-blind review]},
  author={Anonymous Authors},
  year={2026}
}
```

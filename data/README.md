# Data

To avoid redistributing patient and clinical images, no datasets are included in
this repository. Please download each dataset from its original source and place
the files in the directory structures described below. The training scripts
expect these locations when loading the data.

---

## PAD-UFES-20

PAD-UFES-20 contains 2,298 clinical skin-lesion images collected using
smartphones and covers six diagnostic categories:

- ACK
- BCC
- MEL
- NEV
- SCC
- SEK

### Directory Structure

```text
data/pad20/
├── metadata.csv
└── images/
```

---

## HAM10000

HAM10000 contains 10,015 dermatoscopic images of pigmented skin lesions and
covers seven diagnostic categories:

- akiec
- bcc
- bkl
- df
- mel
- nv
- vasc

### Directory Structure

```text
data/ham10000/
├── metadata.csv
└── images/
```

---

## ISIC-2020

ISIC-2020 (SIIM-ISIC Melanoma Classification Challenge) contains 33,126
dermoscopic images from over 2,000 patients. Unlike PAD-UFES-20 and
HAM10000, most images lack a specific diagnosis label; the task is binary
melanoma detection using the `benign_malignant` / `target` column instead.

### Directory Structure

```text
data/isic2020/
├── metadata.csv
└── images/
```

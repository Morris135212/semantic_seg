# semantic segmentation

## Abstract
This repo is used for semantic segmentation experiments. The datasets provide the case where there are stamp on it.

## Install requirements and demo
0. clone the repository
```
git clone https://github.com/Morris135212/semantic_seg.git
```
1. Install the requirements. Check on requirements.txt
```
pip install requirements.txt
```
2. Prepare dataset and put it into ```\input```
3. Run demo demo.ipynb

## Demo file
Here you need to change the path of directory where you put the scan images and ground truth.
```python
SCANS_DIR = "input/scans/scans/"
TRUTH_DIR = "input/ground-truth-pixel/ground-truth-pixel/"
```


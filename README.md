# social-stgcnn-contrastive
Social-STGCNN with contrastive learning for trajectory prediction in human crowds.

# Environment

| Component     | Spec                          |
|---------------|-------------------------------|
| GPU           | NVIDIA RTX 2080 Ti            |
| CPU           | 12 cores                      |
| RAM           | 43 GB                         |
| OS            | Ubuntu 18.04                  |
| Python        | 3.8                           |
| PyTorch       | 1.8.1                         |
| CUDA          | 11.1                          |




#Datasets Included

## 1. ETH Dataset

- **Sub-datasets**: `eth`, `hotel`
- **Description**: Collected in outdoor public spaces with natural human crowd behavior.
- 📎 **Download**: [https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz](https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz)
- 🔗 **Official site**: [https://icu.ee.ethz.ch/research/datsets.html](https://icu.ee.ethz.ch/research/datsets.html)

## 2. UCY Dataset

- **Sub-datasets**: `zara1`, `zara2`, `univ`
- **Description**: Annotated pedestrian trajectories collected in a university campus setting.
- 📎 **Download**: [https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data)
- 🔗 **Dataset info**: [https://paperswithcode.com/dataset/ucy](https://paperswithcode.com/dataset/ucy)


# Quick Start
```bash
pip install -r requirements.txt
python train.py
python test.py

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

To quickly set up the environment:

```bash
conda create -n stgcnn python=3.8
conda activate stgcnn
pip install -r requirements.txt

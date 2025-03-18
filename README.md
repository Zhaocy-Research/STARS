# STARS
**Decoding Spatial Transcriptomics at Any Resolution: From Multicellular or Subcellular Spots to Individual Cells**

We presented STARS (Spatial Transcriptomics across Any Resolution for Single Cells). Leveraging Vision Transformer model and contrastive learning, STARS combines high-resolution histology images with spot-level transcriptomics data to decode true single-cell gene expression from any multicellular or subcellular platforms. We demonstrated the advantage of our true single-cell method using public datasets and in-house datasets of mouse lung from 3 ST platforms (Visium, Visium HD and Stereo-seq). STARS was applied at tissue, individual cell, and molecular levels.

Framework

![image](https://github.com/Zhaocy-Research/STARS/blob/main/STARS.png)

The code is licensed under the MIT license.




**1. Requirements**

**1.1 Operating systems:**

The code in python has been tested on Linux (Ubuntu 20.04.1 LTS).  

**1.2 Required packages in python:**

anndata   
numpy  
opencv-python   
pandas  
python-louvain  
rpy2  
scanpy  
scipy  
seaborn   
torch  
torch-geometric    
torchvision  
tqdm  
umap-learn  

**1.3 How to install STARS:**  
Before installing STARS, ensure that you have **StarDist** installed in your environment. If not, please follow the [installation instructions here](https://github.com/stardist/stardist).

To download **STARS**, use the following command:

```bash
git clone https://github.com/Zhaocy-Research/STARS.git
```

```
(1) cd STARS

(2) conda create --name STARS python=3.9

(3) conda activate STARS  
```
STARS can be installed via pip using the following command:

```bash
pip install stars-omics
```
After installation, you can import the package in Python as:
```
import stars_omics
```

## 2. Instructions: Demo on mouse lung data.

## We provide an example notebook, **visium_06.ipynb**, to implement the experimental results from the paper.

## Data Access
https://pitt-my.sharepoint.com/:f:/g/personal/chz113_pitt_edu/EuGVB7q_xG1FtaTGj7PrW2wBNVADPt_9ZBJGOxnu0zdMwg?e=X1XMC1

Data can be accessed via the following link. Please download the data and update the data path in the code. The file name remains unchanged. For example, modify the code from:

```python
img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
```

to

```python
img_fold = os.path.join('/your/local/path/', name, '.TIF')
```

*Note:* The path `/ix1/wchen//spatial/Lung/Lung_new/SpaceRanger_output/` is just an example. Other link paths used in the code should also be modified to your local path in the same manner.






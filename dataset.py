import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
from contour_util import *
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from scipy.sparse import isspmatrix
from utils import get_data
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import cv2
import json
from sklearn.preprocessing import LabelEncoder
# import mnnpy
import seaborn as sns
from skimage.measure import label
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from hipt_model_utils import eval_transforms
BCELL = ['CD19', 'CD79A', 'CD79B', 'MS4A1']
TUMOR = ['FASN']
CD4T = ['CD4']
CD8T = ['CD8A', 'CD8B']
DC = ['CLIC2', 'CLEC10A', 'CD1B', 'CD1A', 'CD1E']
MDC = ['LAMP3']
CMM = ['BRAF', 'KRAS']
IG = {'B_cell':BCELL, 'Tumor':TUMOR, 'CD4+T_cell':CD4T, 'CD8+T_cell':CD8T, 'Dendritic_cells':DC, 
        'Mature_dendritic_cells':MDC, 'Cutaneous_Malignant_Melanoma':CMM}
MARKERS = []
for i in IG.values():
    MARKERS+=i
LYM = {'B_cell':BCELL, 'CD4+T_cell':CD4T, 'CD8+T_cell':CD8T}
class LUNG_HD(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False,sp=False,fold=0):
        super(LUNG_HD, self).__init__()
        self.r = 16//2
        self.label_encoder = LabelEncoder()  # Initialize label encoder
        #self.image_features= np.load('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/large_image_features.npz')
        self.train = train
        self.sr = sr
        self.sp=sp
        #names = ['05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        names = ['Visium_HD_Human_Lung_Cancer_tissue_image']
        #names = ['A1','A2','A3','A4','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        #names = ['A1','A2','A3','A4']
        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        # if sr == True:
        #     self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.exp_dict = {}
        # self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name,gene_list)
            # print(spatial_data.shape,'shape')
            self.exp_dict[name] = expression_data
            # self.loc_dict[name] = spatial_data[:, 0:2]  # Storing first two dimensions in self.pos
            self.center_dict[name] = spatial_data[:, 0:2]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list
        #self.label_dict={i: self.get_lbl(i) for i in names}
        #converted_label_dict, label_to_int = self.convert_labels_to_integers(self.label_dict)
        #self.label_dict=converted_label_dict
        # Optionally, save the mapping for future reference
        #with open('label_to_int_mapping.json', 'w') as f:
        #    json.dump(label_to_int, f)

        self.id2name = dict(enumerate(names))

        # self.transforms = transforms.Compose([
        #     transforms.ColorJitter(0.5, 0.5, 0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(degrees=180),
        #     transforms.ToTensor()
        # ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def check_tensor(self,tensor, name):
        print(f"{name} dtype: {tensor.dtype}")
        print(f"{name} device: {tensor.device}")
        print(f"{name} contains NaN: {torch.isnan(tensor).any().item()}")
        print(f"{name} contains Inf: {torch.isinf(tensor).any().item()}")
        print(f"{name} min value: {tensor.min().item()}")
        print(f"{name} max value: {tensor.max().item()}")

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #im_patch=self.image_features[i]
        #im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        # centers_org = np.asarray(centers)
        # loc = self.loc_dict[self.id2name[i]]
        # positions = torch.LongTensor(loc)
        #patch_dim = 3 * self.r * self.r * 4
        patch_size=self.r *2
        #label_spot=self.label_dict[self.id2name[i]]

        if self.sr:
            patches, centers = cut_into_overlapping_patches(im, (112, 112),64)
            return patches,  centers, label_spot, im_torch.shape

        elif self.sp:
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            #patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
            new_centers = []  # List to store new centers
            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = normalized_patch
                new_centers.append([x, y])
            return patches, exps, torch.Tensor(new_centers), im_torch.shape
        else:
            n_patches = len(centers)
            # print(len(centers_org))
            #patch_dim = (384)
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            #print(im_torch.shape,'shape2')
            new_centers = [] 
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                #print(f"x: {x}, type: {type(x)}")
                #print(f"y: {y}, type: {type(y)}")
                #print(f"self.r: {self.r}, type: {type(self.r)}")
                x_start = max(0, x - self.r)
                x_end = min(im_torch.shape[0], x + self.r)
                y_start = max(0, y - self.r)
                y_end = min(im_torch.shape[1], y + self.r)
                # Check if any index goes beyond the image boundary
                if x < 0 or y < 0:
                    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x +self.r > im_torch.shape[0] or y + self.r > im_torch.shape[1]:
                #    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x_start >= im_torch.shape[0] or x_end > im_torch.shape[0] or y_start >= im_torch.shape[1] or y_end > im_torch.shape[1]:
                #    print(f"Boundary exceeded for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #    continue  # Skip this patch
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                #normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = patch
                new_centers.append([x, y])
            if self.train:
                return patches, exps,torch.Tensor(new_centers)
            else:
                return patches,  exps,torch.Tensor(new_centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(name):
        if name in ['A1', 'A2', 'A3','A4']:
            img_fold = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name,
                                'outs/spatial/full_image.tif')
        elif name=='08_WT_F_S':
            name='08_WT_F-S'
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        elif name=='09_WT_F_S':
            name='09_WT_F-S'
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        else:
            img_fold = '/ix1/wchen/liutianhao/data/ZhiyuDai/VisiumHD_mouse/HD_WT/Images/CAVG10675_2024-05-03_15-50-08_2024-05-03_15-26-10_H1-MFCPRKP_D1_HD.tif'
        print(os.path.exists(img_fold))
        img_color = cv2.imread(img_fold, cv2.IMREAD_UNCHANGED)
        img_color = img_color.astype(np.float32) / 255.0
        img_color = eval_transforms()(img_color)
        img_color=img_color.permute(1, 2, 0)
        #print(img_color.shape,'shape')
        # img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        #print(img_color.shape,'shape1')
        return img_color
    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1


        return torch.from_numpy(final_labels)

    def convert_labels_to_integers(self, label_dict):
        """
        Convert string labels to integers in label_dict and ensure consistency across samples.

        Parameters:
            label_dict (dict): A dictionary where keys are sample names and values are lists of string labels.

        Returns:
            converted_label_dict (dict): A dictionary with string labels converted to integers.
            label_to_int (dict): A mapping from string labels to integers.
        """

        # Gather all unique labels across all samples
        all_labels = set()
        for labels in label_dict.values():
            all_labels.update(set(labels))

        # Create a mapping from label names to integers
        label_to_int = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}

        # Convert string labels to integers and create tensors
        converted_label_dict = {}
        for sample, labels in label_dict.items():
            converted_labels = [label_to_int[label] for label in labels]
            converted_label_dict[sample] = torch.tensor(converted_labels, dtype=torch.long)

        return converted_label_dict, label_to_int
    def value_binning(self,adata, B=10):
        # Function to bin values for a single spot or cell
        def bin_values(spot_or_cell_values, B):
            # Remove zero values and calculate bin edges
            non_zero_values = spot_or_cell_values[spot_or_cell_values > 0]
            if len(non_zero_values) == 0:
                return np.zeros(spot_or_cell_values.shape)
        
            # Calculate bin edges based on non-zero values in the current spot or cell
            bin_edges = np.percentile(non_zero_values, np.linspace(0, 100, B + 1))
            bin_edges[-1] = bin_edges[-1] + 1  # ensure the max value is included in the last bin

            # Bin the values
            binned_values = np.digitize(spot_or_cell_values, bins=bin_edges) - 1
            binned_values[spot_or_cell_values == 0] = 0  # retain zero values as zero

            return binned_values
    
        # Preprocessing: log1p transformation and HVG selection
        # sc.pp.log1p(adata)
        # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        # adata = adata[:, adata.var['highly_variable']]
    
        # Apply binning for each spot or cell
        binned_matrix = np.apply_along_axis(bin_values, 1, adata.X.toarray(), B)
    
        # Store the binned values back in adata object
        adata.X= binned_matrix
    
        return adata
    def get_cnt(self, name,hvg_list):
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        elif name=='05_WT_SA_Only':
            name='05_WT_SA_Only_2'
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        elif name=='03_WT_SA_Only':
            name='03_WT_SA_Only_2'
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/liutianhao/data/ZhiyuDai/VisiumHD_mouse/HD_WT/outs/binned_outputs/', 'square_008um/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        adata = adata[:, adata.var_names.isin(hvg_list)]
        #sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        #sc.pp.normalize_total(adata, target_sum=1e4)
        #sc.pp.log1p(adata)
        #sc.pp.scale(adata)
        adata = adata[:, hvg_list]
        #adata = self.value_binning(adata, B=20)
        data = adata.X
        if isspmatrix(data):  
            data = data.toarray()
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        adata.X = (data - data_min) / (data_max - data_min+1e-12)

        # Handle division by zero if any max is equal to min
        #adata.X[np.isnan(adata.X)] = 0
        #print(adata,'adata')
        if name in ['A1', 'A2', 'A3','A4']:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        elif name=='05_WT_SA_Only':
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        elif name=='03_WT_SA_Only':
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        else:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions.parquet")
        if name in ['A1', 'A2', 'A3','A4']:
            positions = pd.read_csv(file_Adj, header=None)
            positions.columns = [
                'barcode',
                'in_tissue',
                'array_row',
                'array_col',
                'pxl_row_in_fullres',
                'pxl_col_in_fullres',
        ]
        else:
            positions = pd.read_parquet(file_Adj)
        positions = positions[positions['in_tissue'] == 1]
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)
        #merged_obs = adata.obs.join(positions, how='inner', lsuffix='_adata', rsuffix='_positions')
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        #adata.obsm['spatial'] = merged_obs[[ 'pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        spatial_data = adata.obsm['spatial']
        #filter_mask = (spatial_data[:, 0] >= 128) & (spatial_data[:, 1] >= 128)
        #filtered_spatial_data = spatial_data[filter_mask]

        # Update adata.obsm['spatial'] with the filtered data
        #adata = adata[filter_mask]
        # Identify overlapping columns
        #overlap_columns = adata.obs.columns.intersection(positions.columns)
        
        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        #for col in overlap_columns:
        #    adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        #non_overlap_columns = positions.columns.difference(adata.obs.columns)
        #adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        #adata.obs = adata.obs.merge(positions, left_index=True, right_index=True, how='left')
        #adata.obsm['spatial'] = adata.obs[['array_row', 'array_col','pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        expression_data = adata.X
        if hasattr(expression_data, 'todense'):
            expression_data = expression_data.todense()
        #expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        #data_min = torch.min(expression_data, dim=0)[0]  # min per column
        #data_max = torch.max(expression_data, dim=0)[0]  # max per column

        # Perform min-max normalization
        #expression_data = (expression_data - data_min) / (data_max - data_min)

        # Handle potential NaN values if data_max == data_min
        #expression_data[expression_data != expression_data] = 0  # Replace NaN with 0
        spatial_data = adata.obsm['spatial']
        #print(spatial_data.shape)
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data,gene_list
        # return adata

    def get_pos(self, name):

        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        path = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/labels/', name+'_label.csv')
        # print(path)
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        elif name=='05_WT_SA_Only':
            name='05_WT_SA_Only_2'
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        elif name=='03_WT_SA_Only':
            name='03_WT_SA_Only_2'
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        if os.path.exists(path):
            label_data = pd.read_csv(path, sep=',')
            # print(label_data)
            label_data.columns = ['ID', 'Label']
            aligned_labels = label_data.set_index('ID').reindex(adata.obs_names)
            aligned_labels = aligned_labels.fillna('Background')
            # Add to adata as needed
            # adata.obs['integer_labels'] = aligned_labels
            # integer_labels_tensor = torch.tensor(adata.obs['integer_labels'].values, dtype=torch.int64)

            return aligned_labels
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class SKIN_single_large(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False,sp=False,fold=0):
        super(SKIN_single_large, self).__init__()
        self.r = 256 // 2
        self.label_encoder = LabelEncoder()  # Initialize label encoder
        #self.image_features= np.load('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/large_image_features.npz')
        self.train = train
        self.sr = sr
        self.sp=sp
        #names = ['05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        #names = ['A1','A2','A3','A4','05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        names = ['A1_aligned','B1_aligned','C1_aligned','D1_aligned']
        #names = ['A1','A2','A3','A4']
        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        # if sr == True:
        #     self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.exp_dict = {}
        # self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name,gene_list)
            # print(spatial_data.shape,'shape')
            self.exp_dict[name] = expression_data
            # self.loc_dict[name] = spatial_data[:, 0:2]  # Storing first two dimensions in self.pos
            self.center_dict[name] = spatial_data[:, 2:4]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list
        #self.label_dict={i: self.get_lbl(i) for i in names}
        #converted_label_dict, label_to_int = self.convert_labels_to_integers(self.label_dict)
        #self.label_dict=converted_label_dict
        # Optionally, save the mapping for future reference
        #with open('label_to_int_mapping.json', 'w') as f:
        #    json.dump(label_to_int, f)

        self.id2name = dict(enumerate(names))

        # self.transforms = transforms.Compose([
        #     transforms.ColorJitter(0.5, 0.5, 0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(degrees=180),
        #     transforms.ToTensor()
        # ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def check_tensor(self,tensor, name):
        print(f"{name} dtype: {tensor.dtype}")
        print(f"{name} device: {tensor.device}")
        print(f"{name} contains NaN: {torch.isnan(tensor).any().item()}")
        print(f"{name} contains Inf: {torch.isinf(tensor).any().item()}")
        print(f"{name} min value: {tensor.min().item()}")
        print(f"{name} max value: {tensor.max().item()}")

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #im_patch=self.image_features[i]
        #im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        # centers_org = np.asarray(centers)
        # loc = self.loc_dict[self.id2name[i]]
        # positions = torch.LongTensor(loc)
        #patch_dim = 3 * self.r * self.r * 4
        patch_size=self.r *2
        #label_spot=self.label_dict[self.id2name[i]]

        if self.sr:
            patches, centers = cut_into_overlapping_patches(im, (112, 112),64)
            return patches,  centers, patches, im_torch.shape

        elif self.sp:
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            #patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
            new_centers = []  # List to store new centers
            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = normalized_patch
                new_centers.append([x, y])
            return patches, exps, torch.Tensor(new_centers), patches, im_torch.shape
        else:
            n_patches = len(centers)
            # print(len(centers_org))
            #patch_dim = (384)
            patch_dim = (patch_size, patch_size, 3)
            filtered_centers = []
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                if x >= 0 and y >= 0:
                    filtered_centers.append(center)
            #n_patches = len(filtered_centers)
            #print(n_patches)
            #patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            #print(im_torch.shape,'shape2')
            new_centers = [] 
            valid_exps=[]
            patches=[]
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                if x < 0 or y < 0:
                    #print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                    continue
                #print(f"x: {x}, type: {type(x)}")
                #print(f"y: {y}, type: {type(y)}")
                #print(f"self.r: {self.r}, type: {type(self.r)}")
                x_start = max(0, x - self.r)
                x_end = min(im_torch.shape[0], x + self.r)
                y_start = max(0, y - self.r)
                y_end = min(im_torch.shape[1], y + self.r)
                # Check if any index goes beyond the image boundary
                #if x < 0 or y < 0:
                #    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x +self.r > im_torch.shape[0] or y + self.r > im_torch.shape[1]:
                #    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                if x_start >= im_torch.shape[0] or x_end > im_torch.shape[0] or y_start >= im_torch.shape[1] or y_end > im_torch.shape[1]:
                    #print(f"Boundary exceeded for center {i}: x = {x}, y = {y}, r = {self.r}")
                    #print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                    continue  # Skip this patch
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                #normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                #patches[i, :, :, :] = patch
                patches.append(patch)
                new_centers.append([x, y])
                valid_exps.append(exps[i])
            valid_exps = torch.stack(valid_exps) 
            patches = torch.stack(patches)
            if self.train:
                return patches,valid_exps,valid_exps,torch.Tensor(new_centers)
            else:
                return patches,valid_exps,valid_exps,torch.Tensor(new_centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        if name=='A1_aligned':
            name='S-16'
            img_fold = os.path.join('/ix1/wchen/zhongli/RWorkSpace/Spatial/Skin/ST_v2/data/EVOS_10x_Images/', name + '.TIF')
        elif name=='B1_aligned':
            name='660'
            img_fold = os.path.join('/ix1/wchen/zhongli/RWorkSpace/Spatial/Skin/ST_v2/data/EVOS_10x_Images/', name + '.TIF')
        elif name=='C1_aligned':
            name='437'
            img_fold = os.path.join('/ix1/wchen/zhongli/RWorkSpace/Spatial/Skin/ST_v2/data/EVOS_10x_Images/', name + '.TIF')
        elif name=='D1_aligned':
            name='488'
            img_fold = os.path.join('/ix1/wchen/zhongli/RWorkSpace/Spatial/Skin/ST_v2/data/EVOS_10x_Images/', name + '.TIF')
        else:
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        #print(img_color.shape,'shape')
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img_color = img_color.astype(np.float32) / 255.0
        img_color = eval_transforms()(img_color)
        img_color=img_color.permute(1, 2, 0)
        #print(img_color.shape,'shape1')
        return img_color

    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1


        return torch.from_numpy(final_labels)

    def convert_labels_to_integers(self, label_dict):
        """
        Convert string labels to integers in label_dict and ensure consistency across samples.

        Parameters:
            label_dict (dict): A dictionary where keys are sample names and values are lists of string labels.

        Returns:
            converted_label_dict (dict): A dictionary with string labels converted to integers.
            label_to_int (dict): A mapping from string labels to integers.
        """

        # Gather all unique labels across all samples
        all_labels = set()
        for labels in label_dict.values():
            all_labels.update(set(labels))

        # Create a mapping from label names to integers
        label_to_int = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}

        # Convert string labels to integers and create tensors
        converted_label_dict = {}
        for sample, labels in label_dict.items():
            converted_labels = [label_to_int[label] for label in labels]
            converted_label_dict[sample] = torch.tensor(converted_labels, dtype=torch.long)

        return converted_label_dict, label_to_int

    def get_cnt(self, name,hvg_list):
        if name in ['A1_aligned','B1_aligned','C1_aligned','D1_aligned']:
            input_dir = os.path.join('/ix1/wchen/zhongli/RWorkSpace/Spatial/Skin/ST_v2/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        adata = adata[:, adata.var_names.isin(hvg_list)]
        #sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        #sc.pp.normalize_total(adata, target_sum=1e4)
        #sc.pp.log1p(adata)
        #sc.pp.scale(adata)
        adata = adata[:, hvg_list]
        data = adata.X
        if isspmatrix(data):  
            data = data.toarray()
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        adata.X = (data - data_min) / (data_max - data_min)

        # Handle division by zero if any max is equal to min
        #adata.X[np.isnan(adata.X)] = 0
        #print(adata,'adata')
        if name in ['A1_aligned','B1_aligned','C1_aligned','D1_aligned']:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        else:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        if name in ['A1_aligned','B1_aligned','C1_aligned','D1_aligned']:
            positions = pd.read_csv(file_Adj, header=None)
            positions.columns = [
                'barcode',
                'in_tissue',
                'array_row',
                'array_col',
                'pxl_row_in_fullres',
                'pxl_col_in_fullres',
        ]
        else:
            positions = pd.read_csv(file_Adj)
        
        # Set the index to barcode for merging
        #positions = positions[(positions['pxl_row_in_fullres'] >= 0) & (positions['pxl_col_in_fullres'] >= 0)]
        positions.set_index('barcode', inplace=True)

        # Identify overlapping columns
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[['array_row', 'array_col','pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        expression_data = adata.X
        if hasattr(expression_data, 'todense'):
            expression_data = expression_data.todense()
        #expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        #data_min = torch.min(expression_data, dim=0)[0]  # min per column
        #data_max = torch.max(expression_data, dim=0)[0]  # max per column

        # Perform min-max normalization
        #expression_data = (expression_data - data_min) / (data_max - data_min)

        # Handle potential NaN values if data_max == data_min
        #expression_data[expression_data != expression_data] = 0  # Replace NaN with 0
        spatial_data = adata.obsm['spatial']
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data,gene_list
        # return adata

    def get_pos(self, name):

        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        path = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/labels/', name+'_label.csv')
        # print(path)
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        if os.path.exists(path):
            label_data = pd.read_csv(path, sep=',')
            # print(label_data)
            label_data.columns = ['ID', 'Label']
            aligned_labels = label_data.set_index('ID').reindex(adata.obs_names)
            aligned_labels = aligned_labels.fillna('Background')
            # Add to adata as needed
            # adata.obs['integer_labels'] = aligned_labels
            # integer_labels_tensor = torch.tensor(adata.obs['integer_labels'].values, dtype=torch.int64)

            return aligned_labels
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class STDataset(torch.utils.data.Dataset):
    """Some Information about STDataset"""
    def __init__(self, adata, img_path, diameter=177.5, train=True):
        super(STDataset, self).__init__()

        self.exp = adata.X.toarray()
        self.im = read_tiff(img_path)
        self.r = np.ceil(diameter/2).astype(int)
        self.train = train
        # self.d_spot = self.d_spot if self.d_spot%2==0 else self.d_spot+1
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.centers = adata.obsm['spatial']
        self.pos = adata.obsm['position_norm']
    def __getitem__(self, index):
        exp = self.exp[index]
        center = self.centers[index]
        x, y = center
        patch = self.im.crop((x-self.r, y-self.r, x+self.r, y+self.r))
        exp = torch.Tensor(exp)
        mask = exp!=0
        mask = mask.float()
        if self.train:
            patch = self.transforms(patch)
        pos = torch.Tensor(self.pos[index])
        return patch, pos, exp, mask

    def __len__(self):
        return len(self.centers)

def cut_into_patches_segmentation(image, seg_task,patch_size):
    """
    Cut image into non-overlapping patches.

    Parameters:
        image (np.array): The original image of shape (H, W, C).
        patch_size (tuple): The size of the patches (patch_h, patch_w).

    Returns:
        patches (np.array): The image patches of shape (num_patches, patch_h, patch_w, C).
        centers (np.array): The center coordinates of each patch of shape (num_patches, 2).
    """
    H, W, C = image.shape
    H, W = seg_task.shape
    patch_h, patch_w = patch_size

    patches = []
    patches_seg = []
    centers = []

    for i in range(0, H, patch_h):
        for j in range(0, W, patch_w):
            # Extract patch
            patch = image[i:i + patch_h, j:j + patch_w, :]
            flattened_patch = patch.reshape(-1)  # Flatten the patch
            patches.append(flattened_patch)
            # patches.append(patch)
            patch_seg=seg_task[i:i + patch_h, j:j + patch_w]
            patches_seg.append(patch_seg)
            # Save center
            center = [j + patch_w // 2, i + patch_h // 2]
            centers.append(center)

    return np.array(patches), np.array(centers),np.array(patches_seg)
def cut_into_patches_segmentation_lung(image,patch_size):
    """
    Cut image into non-overlapping patches.

    Parameters:
        image (np.array): The original image of shape (H, W, C).
        patch_size (tuple): The size of the patches (patch_h, patch_w).

    Returns:
        patches (np.array): The image patches of shape (num_patches, patch_h, patch_w, C).
        centers (np.array): The center coordinates of each patch of shape (num_patches, 2).
    """
    H, W, C = image.shape
    # H, W = seg_task.shape
    patch_h, patch_w = patch_size

    patches = []
    # patches_seg = []
    centers = []

    for i in range(0, H, patch_h):
        for j in range(0, W, patch_w):
            # Extract patch
            patch = image[i:i + patch_h, j:j + patch_w, :]
            flattened_patch = patch.reshape(-1)  # Flatten the patch
            patches.append(flattened_patch)
            # patches.append(patch)
            # patch_seg=seg_task[i:i + patch_h, j:j + patch_w]
            # patches_seg.append(patch_seg)
            # Save center
            center = [j + patch_w // 2, i + patch_h // 2]
            centers.append(center)

    return np.array(patches), np.array(centers)

def get_largest_contour(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("binary_mask is empty or None")
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour
def cut_into_overlapping_patches(image, patch_size, stride):
    """
    Cut image into overlapping patches.

    Parameters:
        image (torch.Tensor): The original image of shape (C, H, W).
        patch_size (tuple): The size of the patches (patch_h, patch_w).
        stride (int): The stride of the sliding window.

    Returns:
        patches (torch.Tensor): The image patches.
        positions (torch.Tensor): The (row, col) order of each patch in its block.
        centers (torch.Tensor): The (x, y) coordinates of the center of each patch in the original image.
    """
    # print(image.shape)
    if not torch.is_tensor(image):
        raise TypeError(f"Expected a PyTorch tensor, but got {type(image)}")
    
    # Move to CPU if it's a CUDA tensor
    if image.is_cuda:
        image = image.cpu()
    
    # Convert to NumPy array
    image_np = image.numpy()
    
    # If image is in [C, H, W] format, transpose to [H, W, C]
    if image_np.shape[0] in [1, 3]:  # assuming image has 1 or 3 channels
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Convert to uint8 if it's in [0, 1] range
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    
    # Now you can use image_np with OpenCV functions
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
    gray = gray.astype(np.uint8)
    gray = np.clip(gray, 0, 255)
    H, W,C = image.shape
    patch_h, patch_w = patch_size
    
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary_mask = cv2.threshold(gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
    largest_contour = get_largest_contour(binary_mask)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Slide a window of size patch_size over the region within the bounding box
    patches = []
    # positions = []
    centers = []

    # Slide a window of size patch_size over the image with a step of stride
    for i in range(y, y + h - patch_h + 1, stride):
        for j in range(x, x + w - patch_w + 1, stride):
            # Extract patch
            patch = image[i:i + patch_h, j:j + patch_w,:]
            patches.append(patch)

            # # Save position (order in the block)
            # position = (i // stride, j // stride)
            # positions.append(position)

            # Save center
            center = (j + patch_w // 2, i + patch_h // 2)
            centers.append(center)

    return torch.stack(patches), torch.tensor(centers)

def cut_into_overlapping_patches1(image, patch_size, stride):
    """
    Cut image into overlapping patches.

    Parameters:
        image (torch.Tensor): The original image of shape (C, H, W).
        patch_size (tuple): The size of the patches (patch_h, patch_w).
        stride (int): The stride of the sliding window.

    Returns:
        patches (torch.Tensor): The image patches.
        positions (torch.Tensor): The (row, col) order of each patch in its block.
        centers (torch.Tensor): The (x, y) coordinates of the center of each patch in the original image.
    """
    # print(image.shape)
    H, W,C = image.shape
    patch_h, patch_w = patch_size

    patches = []
    positions = []
    centers = []

    # Slide a window of size patch_size over the image with a step of stride
    for i in range(0, H - patch_h + 1, stride):
        for j in range(0, W - patch_w + 1, stride):
            # Extract patch
            patch = image[i:i + patch_h, j:j + patch_w,:]
            patches.append(patch.flatten())

            # Save position (order in the block)
            position = (i // stride, j // stride)
            positions.append(position)

            # Save center
            center = (j + patch_w // 2, i + patch_h // 2)
            centers.append(center)

    return torch.stack(patches), torch.tensor(positions), torch.tensor(centers)

class HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,fold=0):
        super(HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//4
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        # print(names)
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names
        te_names = [samples[fold]]
        # print(te_names)
        tr_names = list(set(samples)-set(te_names))
        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names
        print('Loading imgs...')
        self.img_dict = {i:self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}

        # self.gene_set = self.get_overlap(self.meta_dict,gene_list)
        # print(len(self.gene_set))
        # np.save('data/her_hvg',self.gene_set)
        # quit()
        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        # if self.cls or self.train==False:

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)
        # print(exp.shape)
        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x-self.r, y-self.r, x+self.r, y+self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else: 
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(ViT_HER2ST, self).__init__()

        self.cnt_dir = '/ix1/wchen/Zhaochongyue/spatial/data/her2st/data/ST-cnts'
        self.img_dir = '/ix1/wchen/Zhaochongyue/spatial/data/her2st/data/ST-imgs'
        self.pos_dir = '/ix1/wchen/Zhaochongyue/spatial/data/her2st/data/ST-spotfiles'
        self.lbl_dir = '/ix1/wchen/Zhaochongyue/spatial/data/her2st/data/ST-pat/lbl'
        self.r = 224//2

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('/ix1/wchen/Zhaochongyue/spatial/data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.sr = sr
        
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        # te_names=[]
        samples = names
        # print(names)
        # te_names = fold
        te_names = [samples[fold]]
        # print(te_names)
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in names}
        if sr==True:
            self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}
        self.cnt_dict={i:self.get_cnt(i) for i in names}
        # print(self.cnt_dict)
        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        # self.exp_dict1 = {i:m[self.gene_set] for i,m in self.meta_dict.items()}
        # corrected = sc.external.pp.mnn_correct(self.exp_dict1,do_concatenate=False)
        # corrected =dict((y, x) for x, y in corrected)
        # print(corrected[0][0])
        # print(self.exp_dict1)
        # self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
        #                  dict(corrected[0][0]).items()}
        # print(corrected[0].shape)
        # print(self.exp_dict1)
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1,0,2)
        patch_size=self.r *2
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        centers_org=np.asarray(centers)
        # print(centers_org.shape)
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        # print(centers[0].shape)
        if self.sr:
            im1 = self.img_dict1[self.id2name[i]]
            # im1 = np.swapaxes(im1,0,1)
            # print(im1.shape)
            # print(im.shape)
            cnt = cv2_detect_contour(im1, apertureSize=5, L2gradient=True)
            binary = np.zeros((im1.shape[0:2]), dtype=np.uint8)
            cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
            # Enlarged filter
            cnt_enlarged = scale_contour(cnt, 1.05)
            binary_enlarged = np.zeros(im1.shape[0:2])
            cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
            # img_new = im.copy()
            # cv2.drawContours(img_new, [cnt], -1, (255), thickness=50)
            # resize_factor = 1000 / np.min(im.shape[0:2])
            # resize_width = int(im.shape[1] * resize_factor)
            # resize_height = int(im.shape[0] * resize_factor)
            # img_new = cv2.resize(img_new, ((resize_width, resize_height)))
            # cv2.imwrite('./cnt.jpg', img_new)
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y
            index=0
            index_list=[]
            # print(binary_enlarged.shape,max_x,max_y)
            while y < max_y:  
                x = min_x            
                while x < max_x:
                    if binary_enlarged[y, x] != 0:
                        centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                        positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            n_patches = len(centers)
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim) 
            for i in range(n_patches):
                # center = centers[i].cpu().numpy()
                # idx=np.where((centers_org[:,0] == center[0]) &(centers_org[:,1] == center[1]))
                # print(len(idx))
                # if i==0:
                #     print(center,centers_org)
                    # center=centers_org[i]
                # if len(idx[0])>0:
                #     # result = np.where((centers_org[0] == center[0]) &(centers_org[1] == center[1]))
                #     # print(center,centers_org[result])
                #     index_list.append(i)
                center=centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i, :, :, :] = patch
            # print(len(index_list))
            # index_list=np.asarray(index_list)
            # print(len(centers_org))
            # np.save('./index_list', index_list)
            # print(patches.shape)
            return patches, positions, centers

        else:    
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim) 
            exps = torch.Tensor(exps)


            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i, :, :, :]=patch

            if self.train:
                return patches, positions, exps
            else: 
                return patches, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name

        im = Image.open(path)
        return im

    def get_img1(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name

        im = cv2.imread(path)
        return im
    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)
class ViT_HER2ST1(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(ViT_HER2ST1, self).__init__()

        self.cnt_dir = '/ix1/wchen/Zhaochongyue/spatial/data/her2st/data/ST-cnts'
        self.img_dir = '/ix1/wchen/Zhaochongyue/spatial/data/her2st/data/ST-imgs'
        self.pos_dir = '/ix1/wchen/Zhaochongyue/spatial/data/her2st/data/ST-spotfiles'
        self.lbl_dir = '/ix1/wchen/Zhaochongyue/spatial/data/her2st/data/ST-pat/lbl'
        self.r = 112//2

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('/ix1/wchen/Zhaochongyue/spatial/data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.sr = sr
        
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        # te_names=[]
        samples = names
        # print(names)
        # te_names = fold
        te_names = [samples[fold]]
        # print(te_names)
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in names}
        if sr==True:
            self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}
        self.cnt_dict={i:self.get_cnt(i) for i in names}
        # print(self.cnt_dict)
        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        # self.exp_dict1 = {i:m[self.gene_set] for i,m in self.meta_dict.items()}
        # corrected = sc.external.pp.mnn_correct(self.exp_dict1,do_concatenate=False)
        # corrected =dict((y, x) for x, y in corrected)
        # print(corrected[0][0])
        # print(self.exp_dict1)
        # self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
        #                  dict(corrected[0][0]).items()}
        # print(corrected[0].shape)
        # print(self.exp_dict1)
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1,0,2)
        patch_size=self.r *2
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        centers_org=np.asarray(centers)
        # print(centers_org.shape)
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        # print(centers[0].shape)
        if self.sr:
            im1 = self.img_dict1[self.id2name[i]]
            # im1 = np.swapaxes(im1,0,1)
            # print(im1.shape)
            # print(im.shape)
            cnt = cv2_detect_contour(im1, apertureSize=5, L2gradient=True)
            binary = np.zeros((im1.shape[0:2]), dtype=np.uint8)
            cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
            # Enlarged filter
            cnt_enlarged = scale_contour(cnt, 1.05)
            binary_enlarged = np.zeros(im1.shape[0:2])
            cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
            # img_new = im.copy()
            # cv2.drawContours(img_new, [cnt], -1, (255), thickness=50)
            # resize_factor = 1000 / np.min(im.shape[0:2])
            # resize_width = int(im.shape[1] * resize_factor)
            # resize_height = int(im.shape[0] * resize_factor)
            # img_new = cv2.resize(img_new, ((resize_width, resize_height)))
            # cv2.imwrite('./cnt.jpg', img_new)
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y
            index=0
            index_list=[]
            # print(binary_enlarged.shape,max_x,max_y)
            while y < max_y:  
                x = min_x            
                while x < max_x:
                    if binary_enlarged[y, x] != 0:
                        centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                        positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                # center = centers[i].cpu().numpy()
                # idx=np.where((centers_org[:,0] == center[0]) &(centers_org[:,1] == center[1]))
                # print(len(idx))
                # if i==0:
                #     print(center,centers_org)
                    # center=centers_org[i]
                # if len(idx[0])>0:
                #     # result = np.where((centers_org[0] == center[0]) &(centers_org[1] == center[1]))
                #     # print(center,centers_org[result])
                #     index_list.append(i)
                center=centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i, :, :, :] = patch.flatten()
            # print(len(index_list))
            # index_list=np.asarray(index_list)
            # print(len(centers_org))
            # np.save('./index_list', index_list)
            # print(patches.shape)
            return patches, positions, centers

        else:    
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = 3 * self.r * self.r * 4
            patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)


            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i, :]=patch.flatten()

            if self.train:
                return patches, positions, exps
            else: 
                return patches, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name

        im = Image.open(path)
        return im

    def get_img1(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name

        im = cv2.imread(path)
        return im
    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)
class LUNG(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False,sp=False,fold=0):
        super(LUNG, self).__init__()
        self.r = 224 // 2
        self.label_encoder = LabelEncoder()  # Initialize label encoder

        self.train = train
        self.sr = sr
        self.sp=sp
        #names = ['05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        #names = ['A1','A2','A3','A4','05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        names = ['A1','A2','A3','A4','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        #names = ['A1','A2','A3','A4']
        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        # if sr == True:
        #     self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.exp_dict = {}
        # self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name)
            # print(spatial_data.shape,'shape')
            self.exp_dict[name] = expression_data
            # self.loc_dict[name] = spatial_data[:, 0:2]  # Storing first two dimensions in self.pos
            self.center_dict[name] = spatial_data[:, 2:4]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list
        self.label_dict={i: self.get_lbl(i) for i in names}
        converted_label_dict, label_to_int = self.convert_labels_to_integers(self.label_dict)
        self.label_dict=converted_label_dict
        # Optionally, save the mapping for future reference
        with open('label_to_int_mapping.json', 'w') as f:
            json.dump(label_to_int, f)

        self.id2name = dict(enumerate(names))

        # self.transforms = transforms.Compose([
        #     transforms.ColorJitter(0.5, 0.5, 0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(degrees=180),
        #     transforms.ToTensor()
        # ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def check_tensor(self,tensor, name):
        print(f"{name} dtype: {tensor.dtype}")
        print(f"{name} device: {tensor.device}")
        print(f"{name} contains NaN: {torch.isnan(tensor).any().item()}")
        print(f"{name} contains Inf: {torch.isinf(tensor).any().item()}")
        print(f"{name} min value: {tensor.min().item()}")
        print(f"{name} max value: {tensor.max().item()}")

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        # centers_org = np.asarray(centers)
        # loc = self.loc_dict[self.id2name[i]]
        # positions = torch.LongTensor(loc)
        #patch_dim = 3 * self.r * self.r * 4
        patch_size=self.r *2
        label_spot=self.label_dict[self.id2name[i]]

        if self.sr:
            patches, centers = cut_into_overlapping_patches(im, (112, 112),64)
            return patches,  centers, label_spot, im_torch.shape

        elif self.sp:
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            #patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
            new_centers = []  # List to store new centers
            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = normalized_patch
                new_centers.append([x, y])
            return patches, exps, torch.Tensor(new_centers), label_spot, im_torch.shape
        else:
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            #patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            #print(im_torch.shape,'shape2')
            new_centers = [] 
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                #print(f"x: {x}, type: {type(x)}")
                #print(f"y: {y}, type: {type(y)}")
                #print(f"self.r: {self.r}, type: {type(self.r)}")
                x_start = max(0, x - self.r)
                x_end = min(im_torch.shape[0], x + self.r)
                y_start = max(0, y - self.r)
                y_end = min(im_torch.shape[1], y + self.r)
                # Check if any index goes beyond the image boundary
                if x < 0 or y < 0:
                    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x +self.r > im_torch.shape[0] or y + self.r > im_torch.shape[1]:
                #    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x_start >= im_torch.shape[0] or x_end > im_torch.shape[0] or y_start >= im_torch.shape[1] or y_end > im_torch.shape[1]:
                #    print(f"Boundary exceeded for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #    continue  # Skip this patch
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = normalized_patch
                new_centers.append([x, y])
            if self.train:
                return patches, exps,label_spot,torch.Tensor(new_centers)
            else:
                return patches,  exps,label_spot,torch.Tensor(new_centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        if name in ['A1', 'A2', 'A3','A4']:
            img_fold = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name,
                                'outs/spatial/full_image.tif')
        elif name=='08_WT_F_S':
            name='08_WT_F-S'
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        elif name=='09_WT_F_S':
            name='09_WT_F-S'
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        else:
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        #print(img_color.shape,'shape')
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        #print(img_color.shape,'shape1')
        return img_color

    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1


        return torch.from_numpy(final_labels)

    def convert_labels_to_integers(self, label_dict):
        """
        Convert string labels to integers in label_dict and ensure consistency across samples.

        Parameters:
            label_dict (dict): A dictionary where keys are sample names and values are lists of string labels.

        Returns:
            converted_label_dict (dict): A dictionary with string labels converted to integers.
            label_to_int (dict): A mapping from string labels to integers.
        """

        # Gather all unique labels across all samples
        all_labels = set()
        for labels in label_dict.values():
            all_labels.update(set(labels))

        # Create a mapping from label names to integers
        label_to_int = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}

        # Convert string labels to integers and create tensors
        converted_label_dict = {}
        for sample, labels in label_dict.items():
            converted_labels = [label_to_int[label] for label in labels]
            converted_label_dict[sample] = torch.tensor(converted_labels, dtype=torch.long)

        return converted_label_dict, label_to_int

    def get_cnt(self, name):
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        #print(adata,'adata')
        if name in ['A1', 'A2', 'A3','A4']:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        else:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        if name in ['A1', 'A2', 'A3','A4']:
            positions = pd.read_csv(file_Adj, header=None)
            positions.columns = [
                'barcode',
                'in_tissue',
                'array_row',
                'array_col',
                'pxl_row_in_fullres',
                'pxl_col_in_fullres',
        ]
        else:
            positions = pd.read_csv(file_Adj)
        
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)

        # Identify overlapping columns
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[['array_row', 'array_col','pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        expression_data = adata.X
        expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        spatial_data = adata.obsm['spatial']
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data,gene_list
        # return adata

    def get_pos(self, name):

        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        path = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/labels/', name+'_label.csv')
        # print(path)
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        if os.path.exists(path):
            label_data = pd.read_csv(path, sep=',')
            # print(label_data)
            label_data.columns = ['ID', 'Label']
            aligned_labels = label_data.set_index('ID').reindex(adata.obs_names)
            aligned_labels = aligned_labels.fillna('Background')
            # Add to adata as needed
            # adata.obs['integer_labels'] = aligned_labels
            # integer_labels_tensor = torch.tensor(adata.obs['integer_labels'].values, dtype=torch.int64)

            return aligned_labels
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class LUNG_single(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False,sp=False,fold=0):
        super(LUNG_single, self).__init__()
        self.r = 224 // 2
        self.label_encoder = LabelEncoder()  # Initialize label encoder

        self.train = train
        self.sr = sr
        self.sp=sp
        #names = ['05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        #names = ['A1','A2','A3','A4','05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        names = ['A1','A2','A3','A4','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        #names = ['A1','A2','A3','A4']
        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        # if sr == True:
        #     self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.exp_dict = {}
        # self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name,gene_list)
            # print(spatial_data.shape,'shape')
            self.exp_dict[name] = expression_data
            # self.loc_dict[name] = spatial_data[:, 0:2]  # Storing first two dimensions in self.pos
            self.center_dict[name] = spatial_data[:, 2:4]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list
        self.label_dict={i: self.get_lbl(i) for i in names}
        converted_label_dict, label_to_int = self.convert_labels_to_integers(self.label_dict)
        self.label_dict=converted_label_dict
        # Optionally, save the mapping for future reference
        with open('label_to_int_mapping.json', 'w') as f:
            json.dump(label_to_int, f)

        self.id2name = dict(enumerate(names))

        # self.transforms = transforms.Compose([
        #     transforms.ColorJitter(0.5, 0.5, 0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(degrees=180),
        #     transforms.ToTensor()
        # ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def check_tensor(self,tensor, name):
        print(f"{name} dtype: {tensor.dtype}")
        print(f"{name} device: {tensor.device}")
        print(f"{name} contains NaN: {torch.isnan(tensor).any().item()}")
        print(f"{name} contains Inf: {torch.isinf(tensor).any().item()}")
        print(f"{name} min value: {tensor.min().item()}")
        print(f"{name} max value: {tensor.max().item()}")

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        # centers_org = np.asarray(centers)
        # loc = self.loc_dict[self.id2name[i]]
        # positions = torch.LongTensor(loc)
        #patch_dim = 3 * self.r * self.r * 4
        patch_size=self.r *2
        label_spot=self.label_dict[self.id2name[i]]

        if self.sr:
            patches, centers = cut_into_overlapping_patches(im, (112, 112),64)
            return patches,  centers, label_spot, im_torch.shape

        elif self.sp:
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            #patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
            new_centers = []  # List to store new centers
            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = normalized_patch
                new_centers.append([x, y])
            return patches, exps, torch.Tensor(new_centers), label_spot, im_torch.shape
        else:
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            #patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            #print(im_torch.shape,'shape2')
            new_centers = [] 
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                #print(f"x: {x}, type: {type(x)}")
                #print(f"y: {y}, type: {type(y)}")
                #print(f"self.r: {self.r}, type: {type(self.r)}")
                x_start = max(0, x - self.r)
                x_end = min(im_torch.shape[0], x + self.r)
                y_start = max(0, y - self.r)
                y_end = min(im_torch.shape[1], y + self.r)
                # Check if any index goes beyond the image boundary
                if x < 0 or y < 0:
                    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x +self.r > im_torch.shape[0] or y + self.r > im_torch.shape[1]:
                #    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x_start >= im_torch.shape[0] or x_end > im_torch.shape[0] or y_start >= im_torch.shape[1] or y_end > im_torch.shape[1]:
                #    print(f"Boundary exceeded for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #    continue  # Skip this patch
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = normalized_patch
                new_centers.append([x, y])
            if self.train:
                return patches, exps,label_spot,torch.Tensor(new_centers)
            else:
                return patches,  exps,label_spot,torch.Tensor(new_centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        if name in ['A1', 'A2', 'A3','A4']:
            img_fold = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name,
                                'outs/spatial/full_image.tif')
        elif name=='08_WT_F_S':
            name='08_WT_F-S'
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        elif name=='09_WT_F_S':
            name='09_WT_F-S'
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        else:
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        #print(img_color.shape,'shape')
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        #print(img_color.shape,'shape1')
        return img_color

    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1


        return torch.from_numpy(final_labels)

    def convert_labels_to_integers(self, label_dict):
        """
        Convert string labels to integers in label_dict and ensure consistency across samples.

        Parameters:
            label_dict (dict): A dictionary where keys are sample names and values are lists of string labels.

        Returns:
            converted_label_dict (dict): A dictionary with string labels converted to integers.
            label_to_int (dict): A mapping from string labels to integers.
        """

        # Gather all unique labels across all samples
        all_labels = set()
        for labels in label_dict.values():
            all_labels.update(set(labels))

        # Create a mapping from label names to integers
        label_to_int = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}

        # Convert string labels to integers and create tensors
        converted_label_dict = {}
        for sample, labels in label_dict.items():
            converted_labels = [label_to_int[label] for label in labels]
            converted_label_dict[sample] = torch.tensor(converted_labels, dtype=torch.long)

        return converted_label_dict, label_to_int

    def get_cnt(self, name,hvg_list):
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        adata = adata[:, adata.var_names.isin(hvg_list)]
        #sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, hvg_list]
        #print(adata,'adata')
        if name in ['A1', 'A2', 'A3','A4']:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        else:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        if name in ['A1', 'A2', 'A3','A4']:
            positions = pd.read_csv(file_Adj, header=None)
            positions.columns = [
                'barcode',
                'in_tissue',
                'array_row',
                'array_col',
                'pxl_row_in_fullres',
                'pxl_col_in_fullres',
        ]
        else:
            positions = pd.read_csv(file_Adj)
        
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)

        # Identify overlapping columns
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[['array_row', 'array_col','pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        expression_data = adata.X
        expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        spatial_data = adata.obsm['spatial']
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data,gene_list
        # return adata

    def get_pos(self, name):

        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        path = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/labels/', name+'_label.csv')
        # print(path)
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        if os.path.exists(path):
            label_data = pd.read_csv(path, sep=',')
            # print(label_data)
            label_data.columns = ['ID', 'Label']
            aligned_labels = label_data.set_index('ID').reindex(adata.obs_names)
            aligned_labels = aligned_labels.fillna('Background')
            # Add to adata as needed
            # adata.obs['integer_labels'] = aligned_labels
            # integer_labels_tensor = torch.tensor(adata.obs['integer_labels'].values, dtype=torch.int64)

            return aligned_labels
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)

class LUNG_single_large(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False,sp=False,fold=0):
        super(LUNG_single_large, self).__init__()
        self.r = 256 // 2
        self.label_encoder = LabelEncoder()  # Initialize label encoder
        #self.image_features= np.load('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/large_image_features.npz')
        self.train = train
        self.sr = sr
        self.sp=sp
        #names = ['05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        names = ['05_WT_SA_Only','03_WT_SA_Only','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        #names = ['09_WT_F_S']
        #names = ['A1','A2','A3','A4','01_WT_Naive','08_WT_F_S','07_WT_Flu_Only','02_WT_Naive','09_WT_F_S','06_WT_Flu_Only']
        #names = ['A1','A2','A3','A4']
        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        # if sr == True:
        #     self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.exp_dict = {}
        # self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name,gene_list)
            # print(spatial_data.shape,'shape')
            self.exp_dict[name] = expression_data
            # self.loc_dict[name] = spatial_data[:, 0:2]  # Storing first two dimensions in self.pos
            self.center_dict[name] = spatial_data[:, 0:2]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list
        self.label_dict={i: self.get_lbl(i) for i in names}
        converted_label_dict, label_to_int = self.convert_labels_to_integers(self.label_dict)
        self.label_dict=converted_label_dict
        # Optionally, save the mapping for future reference
        with open('label_to_int_mapping.json', 'w') as f:
            json.dump(label_to_int, f)

        self.id2name = dict(enumerate(names))

        # self.transforms = transforms.Compose([
        #     transforms.ColorJitter(0.5, 0.5, 0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(degrees=180),
        #     transforms.ToTensor()
        # ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])
    def eval_transforms1():
        """
        This function applies normalization to a given torch tensor.
        """
        mean, std = torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])
    
        def normalize(tensor):
            return transforms.functional.normalize(tensor, mean, std)
    
        return normalize
    def check_tensor(self,tensor, name):
        print(f"{name} dtype: {tensor.dtype}")
        print(f"{name} device: {tensor.device}")
        print(f"{name} contains NaN: {torch.isnan(tensor).any().item()}")
        print(f"{name} contains Inf: {torch.isinf(tensor).any().item()}")
        print(f"{name} min value: {tensor.min().item()}")
        print(f"{name} max value: {tensor.max().item()}")

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #im_patch=self.image_features[i]
        #im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        # centers_org = np.asarray(centers)
        # loc = self.loc_dict[self.id2name[i]]
        # positions = torch.LongTensor(loc)
        #patch_dim = 3 * self.r * self.r * 4
        patch_size=self.r *2
        label_spot=self.label_dict[self.id2name[i]]

        if self.sr:
            patches, centers = cut_into_overlapping_patches(im, (112, 112),64)
            return patches,  centers, label_spot, im_torch.shape

        elif self.sp:
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            #patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
            new_centers = []  # List to store new centers
            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = normalized_patch
                new_centers.append([x, y])
            return patches, exps, torch.Tensor(new_centers), label_spot, im_torch.shape
        else:
            n_patches = len(centers)
            # print(len(centers_org))
            #patch_dim = (384)
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            #print(im_torch.shape,'shape2')
            new_centers = [] 
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                #print(f"x: {x}, type: {type(x)}")
                #print(f"y: {y}, type: {type(y)}")
                #print(f"self.r: {self.r}, type: {type(self.r)}")
                x_start = max(0, x - self.r)
                x_end = min(im_torch.shape[0], x + self.r)
                y_start = max(0, y - self.r)
                y_end = min(im_torch.shape[1], y + self.r)
                # Check if any index goes beyond the image boundary
                if x < 0 or y < 0:
                    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x +self.r > im_torch.shape[0] or y + self.r > im_torch.shape[1]:
                #    print(f"Negative index for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #if x_start >= im_torch.shape[0] or x_end > im_torch.shape[0] or y_start >= im_torch.shape[1] or y_end > im_torch.shape[1]:
                #    print(f"Boundary exceeded for center {i}: x = {x}, y = {y}, r = {self.r}")
                #    print(f"Calculated indices: x_start = {x_start}, x_end = {x_end}, y_start = {y_start}, y_end = {y_end}")
                #    continue  # Skip this patch
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                #patch = eval_transforms()(patch)
                #normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = patch
                new_centers.append([x, y])
            if self.train:
                return patches, exps,label_spot,torch.Tensor(new_centers)
            else:
                return patches,  exps,label_spot,torch.Tensor(new_centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        if name in ['A1', 'A2', 'A3','A4']:
            img_fold = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name,
                                'outs/spatial/full_image.tif')
        elif name=='08_WT_F_S':
            name='08_WT_F-S'
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        elif name=='09_WT_F_S':
            name='09_WT_F-S'
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        else:
            img_fold = os.path.join('/ix1/wchen/Shiyue/Projects/2023_06_Influ_Mouse_Lung_ST/RawData/Fastq/Alcorn_Visium_FFPE_Images/', name + '.TIF')
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        #print(img_color.shape,'shape')
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img_color = img_color.astype(np.float32) / 255.0
        img_color = eval_transforms()(img_color)
        img_color=img_color.permute(1, 2, 0)
        #print(img_color.shape,'shape1')
        return img_color

    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1


        return torch.from_numpy(final_labels)

    def convert_labels_to_integers(self, label_dict):
        """
        Convert string labels to integers in label_dict and ensure consistency across samples.

        Parameters:
            label_dict (dict): A dictionary where keys are sample names and values are lists of string labels.

        Returns:
            converted_label_dict (dict): A dictionary with string labels converted to integers.
            label_to_int (dict): A mapping from string labels to integers.
        """

        # Gather all unique labels across all samples
        all_labels = set()
        for labels in label_dict.values():
            all_labels.update(set(labels))

        # Create a mapping from label names to integers
        label_to_int = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}

        # Convert string labels to integers and create tensors
        converted_label_dict = {}
        for sample, labels in label_dict.items():
            converted_labels = [label_to_int[label] for label in labels]
            converted_label_dict[sample] = torch.tensor(converted_labels, dtype=torch.long)

        return converted_label_dict, label_to_int
    def value_binning(self,adata, B=10):
        # Function to bin values for a single spot or cell
        def bin_values(spot_or_cell_values, B):
            # Remove zero values and calculate bin edges
            non_zero_values = spot_or_cell_values[spot_or_cell_values > 0]
            if len(non_zero_values) == 0:
                return np.zeros(spot_or_cell_values.shape)
        
            # Calculate bin edges based on non-zero values in the current spot or cell
            bin_edges = np.percentile(non_zero_values, np.linspace(0, 100, B + 1))
            bin_edges[-1] = bin_edges[-1] + 1  # ensure the max value is included in the last bin

            # Bin the values
            binned_values = np.digitize(spot_or_cell_values, bins=bin_edges) - 1
            binned_values[spot_or_cell_values == 0] = 0  # retain zero values as zero

            return binned_values
    
        # Preprocessing: log1p transformation and HVG selection
        # sc.pp.log1p(adata)
        # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        # adata = adata[:, adata.var['highly_variable']]
    
        # Apply binning for each spot or cell
        binned_matrix = np.apply_along_axis(bin_values, 1, adata.X.toarray(), B)
    
        # Store the binned values back in adata object
        adata.X= binned_matrix
    
        return adata
    def get_cnt(self, name,hvg_list):
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        elif name=='05_WT_SA_Only':
            name='05_WT_SA_Only_2'
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        elif name=='03_WT_SA_Only':
            name='03_WT_SA_Only_2'
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        adata = adata[:, adata.var_names.isin(hvg_list)]
        #sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        #sc.pp.normalize_total(adata, target_sum=1e4)
        #sc.pp.log1p(adata)
        #sc.pp.scale(adata)
        adata = adata[:, hvg_list]
        #sc.pp.normalize_total(adata, target_sum=1e4)
        #sc.pp.log1p(adata)
        #adata = self.value_binning(adata, B=20)
        data = adata.X
        
        #data = adata.X
        if isspmatrix(data):  
            data = data.toarray()
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        #print(data_min.shape,data.shape)
        adata.X = (data - data_min) / (data_max - data_min+1e-12)

        # Handle division by zero if any max is equal to min
        #adata.X[np.isnan(adata.X)] = 0
        #print(adata,'adata')
        if name in ['A1', 'A2', 'A3','A4']:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        elif name=='05_WT_SA_Only':
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        elif name=='03_WT_SA_Only':
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        else:
            file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")
        if name in ['A1', 'A2', 'A3','A4']:
            positions = pd.read_csv(file_Adj, header=None)
            positions.columns = [
                'barcode',
                'in_tissue',
                'array_row',
                'array_col',
                'pxl_row_in_fullres',
                'pxl_col_in_fullres',
        ]
        else:
            positions = pd.read_csv(file_Adj)
        positions = positions[positions['in_tissue'] == 1]
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)
        merged_obs = adata.obs.join(positions, how='inner', lsuffix='_adata', rsuffix='_positions')
        adata.obsm['spatial'] = merged_obs[[ 'pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        spatial_data = adata.obsm['spatial']
        filter_mask = (spatial_data[:, 0] >= 128) & (spatial_data[:, 1] >= 128)
        filtered_spatial_data = spatial_data[filter_mask]

        # Update adata.obsm['spatial'] with the filtered data
        adata = adata[filter_mask]
        # Identify overlapping columns
        #overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        #for col in overlap_columns:
        #    adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        #non_overlap_columns = positions.columns.difference(adata.obs.columns)
        #adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        #adata.obs = adata.obs.merge(positions, left_index=True, right_index=True, how='left')
        #adata.obsm['spatial'] = adata.obs[['array_row', 'array_col','pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        expression_data = adata.X
        if hasattr(expression_data, 'todense'):
            expression_data = expression_data.todense()
        #expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        #data_min = torch.min(expression_data, dim=0)[0]  # min per column
        #data_max = torch.max(expression_data, dim=0)[0]  # max per column

        # Perform min-max normalization
        #expression_data = (expression_data - data_min) / (data_max - data_min)

        # Handle potential NaN values if data_max == data_min
        #expression_data[expression_data != expression_data] = 0  # Replace NaN with 0
        spatial_data = adata.obsm['spatial']
        #print(spatial_data.shape)
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data,gene_list
        # return adata

    def get_pos(self, name):

        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        path = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/labels/', name+'_label.csv')
        # print(path)
        if name in ['A1', 'A2', 'A3','A4']:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        elif name=='05_WT_SA_Only':
            name='05_WT_SA_Only_2'
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        elif name=='03_WT_SA_Only':
            name='03_WT_SA_Only_2'
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        else:
            input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/Lung_new/SpaceRanger_output/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        if os.path.exists(path):
            label_data = pd.read_csv(path, sep=',')
            # print(label_data)
            label_data.columns = ['ID', 'Label']
            aligned_labels = label_data.set_index('ID').reindex(adata.obs_names)
            aligned_labels = aligned_labels.fillna('Background')
            # Add to adata as needed
            # adata.obs['integer_labels'] = aligned_labels
            # integer_labels_tensor = torch.tensor(adata.obs['integer_labels'].values, dtype=torch.int64)

            return aligned_labels
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class LUNG_seg(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(LUNG_seg, self).__init__()

        self.r = 224 // 2
        names = ['A1','A2','A3']
        self.names=names
        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        # if sr == True:
        #     self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.exp_dict = {}
        # self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name)
            self.exp_dict[name] = expression_data
            # self.loc_dict[name] = spatial_data[:, 0:2]  # Storing first two dimensions in self.pos
            self.center_dict[name] = spatial_data[:, 2:4]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list

        self.label_dict = {i: self.get_lbl(i) for i in names}
        # print(self.label_dict)
        converted_label_dict, label_to_int = self.convert_labels_to_integers(self.label_dict)
        # print(converted_label_dict)
        self.label_dict = converted_label_dict
        # Optionally, save the mapping for future reference
        with open('label_to_int_mapping.json', 'w') as f:
            json.dump(label_to_int, f)

        self.train = train
        self.sr = sr

        self.id2name = dict(enumerate(self.names))

        # self.transforms = transforms.Compose([
        #     transforms.ColorJitter(0.5, 0.5, 0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(degrees=180),
        #     transforms.ToTensor()
        # ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def convert_labels_to_integers(self,label_dict):
        try:
            all_labels = set()
            for labels in label_dict.values():
                # Check if labels is a Pandas Series or DataFrame and convert to list if necessary
                if isinstance(labels, (pd.Series, pd.DataFrame)):
                    labels = labels['Label'].tolist()
                # print("Labels:", labels)
                all_labels.update(set(labels))
            # print("All Labels:", all_labels)

            label_to_int = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}
            # print("Label to Int Mapping:", label_to_int)

            converted_label_dict = {}
            for sample, labels in label_dict.items():
                # Check if labels is a Pandas Series or DataFrame and convert to list if necessary
                if isinstance(labels, (pd.Series, pd.DataFrame)):
                    labels = labels['Label'].tolist()
                try:
                    converted_labels = [label_to_int[label] for label in labels]
                except KeyError as e:
                    print(f"Error: Label {str(e)} not found in label_to_int mapping.")
                    return None, None  # or handle as appropriate

                converted_label_dict[sample] = torch.tensor(converted_labels, dtype=torch.long)

            return converted_label_dict, label_to_int
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None, None  # or handle as appropriate

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #im = im.permute(1, 0, 2)

        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        # centers_org = np.asarray(centers)
        # loc = self.loc_dict[self.id2name[i]]
        # positions = torch.LongTensor(loc)
        #patch_dim = 3 * self.r * self.r * 4
        label_spot=self.label_dict[self.id2name[i]]
        patch_size=self.r *2
        if self.sr:
            patches, centers=cut_into_patches_segmentation_lung(im,(112,112))

            return patches, centers,label_spot,img.shape
        else:
            n_patches = len(centers)
            # print(len(centers_org))
            patch_dim = (patch_size, patch_size, 3)
            patches = torch.zeros((n_patches,) + patch_dim)  # Keeping spatial dimensions
            #patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :, :, :] = normalized_patch
            if self.train:

                return patches, exps, label_spot
            else:
                return patches, exps, torch.Tensor(centers), label_spot

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        img_fold = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name,
                                'outs/spatial/full_image.tif')
        # print(img_fold)
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        return img_color

    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1

        return torch.from_numpy(final_labels)

    def get_cnt(self, name):

        input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        # print(input_dir)
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        # print(adata.obs)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_row_in_fullres',
            'pxl_col_in_fullres',
        ]
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)

        # Identify overlapping columns
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[
            ['array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        expression_data = adata.X
        expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        spatial_data = adata.obsm['spatial']
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data, gene_list

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/labels/', name + '_label.csv')
        input_dir = os.path.join('/ix1/wchen/Zhaochongyue/spatial/Lung/ST/', name, 'outs/')
        # print(input_dir)
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        if os.path.exists(path):
            label_data = pd.read_csv(path, sep=',', names=['ID', 'Label'], header=None, encoding='utf-8')
            # print(label_data.head())
            # print(label_data)
            # label_data.columns = ['ID', 'Label']
            aligned_labels = label_data.set_index('ID').reindex(adata.obs_names)
            aligned_labels = aligned_labels.fillna('Background')
            # Add to adata as needed
            # adata.obs['integer_labels'] = aligned_labels
            # integer_labels_tensor = torch.tensor(adata.obs['integer_labels'].values, dtype=torch.int64)

            return aligned_labels
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class LUNG1(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(LUNG1, self).__init__()
        self.r = 224 // 4
        self.label_encoder = LabelEncoder()  # Initialize label encoder

        self.train = train
        self.sr = sr

        names = ['A1','A2']

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        # if sr == True:
        #     self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.exp_dict = {}
        self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name)
            # print(spatial_data.shape,'shape')
            self.exp_dict[name] = expression_data
            self.loc_dict[name] = spatial_data[:, 0:2]  # Storing first two dimensions in self.pos
            self.center_dict[name] = spatial_data[:, 2:4]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list
        self.label_dict={i: self.get_lbl(i) for i in names}
        converted_label_dict, label_to_int = self.convert_labels_to_integers(self.label_dict)
        self.label_dict=converted_label_dict
        # Optionally, save the mapping for future reference
        with open('label_to_int_mapping.json', 'w') as f:
            json.dump(label_to_int, f)

        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def check_tensor(self,tensor, name):
        print(f"{name} dtype: {tensor.dtype}")
        print(f"{name} device: {tensor.device}")
        print(f"{name} contains NaN: {torch.isnan(tensor).any().item()}")
        print(f"{name} contains Inf: {torch.isinf(tensor).any().item()}")
        print(f"{name} min value: {tensor.min().item()}")
        print(f"{name} max value: {tensor.max().item()}")

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        centers_org = np.asarray(centers)
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label_spot=self.label_dict[self.id2name[i]]

        if self.sr:
            patches, centers,positions = cut_into_overlapping_patches1(im, (112, 112),20)

            return patches, positions, centers, label_spot, im.shape

        else:
            n_patches = len(centers)
            # print(len(centers_org))
            patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # Flatten and store the normalized patch
                patches[i, :] = normalized_patch.flatten()
            if self.train:
                return patches, positions, exps,label_spot
            else:
                return patches, positions, exps, torch.Tensor(centers),label_spot

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        img_fold = os.path.join('/media/cyzhao/New_Volume/SEDR-master (1)/data/Lung/', name,
                                'outs/spatial/full_image.tif')
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        return img_color

    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1


        return torch.from_numpy(final_labels)

    def convert_labels_to_integers(self, label_dict):
        """
        Convert string labels to integers in label_dict and ensure consistency across samples.

        Parameters:
            label_dict (dict): A dictionary where keys are sample names and values are lists of string labels.

        Returns:
            converted_label_dict (dict): A dictionary with string labels converted to integers.
            label_to_int (dict): A mapping from string labels to integers.
        """

        # Gather all unique labels across all samples
        all_labels = set()
        for labels in label_dict.values():
            all_labels.update(set(labels))

        # Create a mapping from label names to integers
        label_to_int = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}

        # Convert string labels to integers and create tensors
        converted_label_dict = {}
        for sample, labels in label_dict.items():
            converted_labels = [label_to_int[label] for label in labels]
            converted_label_dict[sample] = torch.tensor(converted_labels, dtype=torch.long)

        return converted_label_dict, label_to_int

    def get_cnt(self, name):
        input_dir = os.path.join('/media/cyzhao/New_Volume/SEDR_analyses-master/data/Lung/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
                'barcode',
                'in_tissue',
                'array_row',
                'array_col',
                'pxl_col_in_fullres',
                'pxl_row_in_fullres',
        ]
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)

        # Identify overlapping columns
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[['array_row', 'array_col','pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy()
        expression_data = adata.X
        expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        spatial_data = adata.obsm['spatial']
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data,gene_list
        # return adata

    def get_pos(self, name):

        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        path = os.path.join('/media/cyzhao/New_Volume/SEDR_analyses-master/data/Lung/labels/', name+'_label.csv')
        # print(path)
        input_dir = os.path.join('/media/cyzhao/New_Volume/SEDR_analyses-master/data/Lung/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        if os.path.exists(path):
            label_data = pd.read_csv(path, sep=',')
            # print(label_data)
            label_data.columns = ['ID', 'Label']
            aligned_labels = label_data.set_index('ID').reindex(adata.obs_names)
            aligned_labels = aligned_labels.fillna('Background')
            # Add to adata as needed
            # adata.obs['integer_labels'] = aligned_labels
            # integer_labels_tensor = torch.tensor(adata.obs['integer_labels'].values, dtype=torch.int64)

            return aligned_labels
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class LUNG_seg1(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(LUNG_seg1, self).__init__()

        self.r = 224 // 4
        names = ['A1','A2']
        self.names=names
        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        # if sr == True:
        #     self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.exp_dict = {}
        self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name)
            self.exp_dict[name] = expression_data
            self.loc_dict[name] = spatial_data[:, 0:2]  # Storing first two dimensions in self.pos
            self.center_dict[name] = spatial_data[:, 2:4]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list

        self.label_dict = {i: self.get_lbl(i) for i in names}
        # print(self.label_dict)
        converted_label_dict, label_to_int = self.convert_labels_to_integers(self.label_dict)
        # print(converted_label_dict)
        self.label_dict = converted_label_dict
        # Optionally, save the mapping for future reference
        with open('label_to_int_mapping.json', 'w') as f:
            json.dump(label_to_int, f)

        self.train = train
        self.sr = sr

        self.id2name = dict(enumerate(self.names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def convert_labels_to_integers(self,label_dict):
        try:
            all_labels = set()
            for labels in label_dict.values():
                # Check if labels is a Pandas Series or DataFrame and convert to list if necessary
                if isinstance(labels, (pd.Series, pd.DataFrame)):
                    labels = labels['Label'].tolist()
                # print("Labels:", labels)
                all_labels.update(set(labels))
            # print("All Labels:", all_labels)

            label_to_int = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}
            # print("Label to Int Mapping:", label_to_int)

            converted_label_dict = {}
            for sample, labels in label_dict.items():
                # Check if labels is a Pandas Series or DataFrame and convert to list if necessary
                if isinstance(labels, (pd.Series, pd.DataFrame)):
                    labels = labels['Label'].tolist()
                try:
                    converted_labels = [label_to_int[label] for label in labels]
                except KeyError as e:
                    print(f"Error: Label {str(e)} not found in label_to_int mapping.")
                    return None, None  # or handle as appropriate

                converted_label_dict[sample] = torch.tensor(converted_labels, dtype=torch.long)

            return converted_label_dict, label_to_int
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None, None  # or handle as appropriate

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)

        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        centers_org = np.asarray(centers)
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label_spot=self.label_dict[self.id2name[i]]

        if self.sr:
            patches, centers=cut_into_patches_segmentation_lung(im,(112,112))

            return patches, positions, centers,label_spot,img.shape
        else:
            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                patches[i, :] = normalized_patch.flatten()
            if self.train:

                return patches, positions, exps, label_spot
            else:
                return patches, positions, exps, torch.Tensor(centers), label_spot

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        img_fold = os.path.join('/media/cyzhao/New_Volume/SEDR-master (1)/data/Lung/', name,
                                'outs/spatial/full_image.tif')
        # print(img_fold)
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        return img_color

    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1

        return torch.from_numpy(final_labels)

    def get_cnt(self, name):

        input_dir = os.path.join('/media/cyzhao/New_Volume/SEDR_analyses-master/data/Lung/', name, 'outs/')
        # print(input_dir)
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        # print(adata.obs)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)

        # Identify overlapping columns
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[
            ['array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy()
        expression_data = adata.X
        expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        spatial_data = adata.obsm['spatial']
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data, gene_list

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = os.path.join('/media/cyzhao/New_Volume/SEDR_analyses-master/data/Lung/labels/', name + '_label.csv')
        input_dir = os.path.join('/media/cyzhao/New_Volume/SEDR_analyses-master/data/Lung/', name, 'outs/')
        # print(input_dir)
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        if os.path.exists(path):
            label_data = pd.read_csv(path, sep=',', names=['ID', 'Label'], header=None, encoding='utf-8')
            # print(label_data.head())
            # print(label_data)
            # label_data.columns = ['ID', 'Label']
            aligned_labels = label_data.set_index('ID').reindex(adata.obs_names)
            aligned_labels = aligned_labels.fillna('Background')
            # Add to adata as needed
            # adata.obs['integer_labels'] = aligned_labels
            # integer_labels_tensor = torch.tensor(adata.obs['integer_labels'].values, dtype=torch.int64)

            return aligned_labels
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class ViT_LUNG(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(ViT_LUNG, self).__init__()

        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.lbl_img = 'data/her2st/data/ST-pat/img'
        self.r = 224 // 4
        self.label_encoder = LabelEncoder()  # Initialize label encoder
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.sr = sr

        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        # te_names=[]
        samples = names
        # print(names)
        # te_names = fold
        te_names = [samples[fold]]
        # print(te_names)
        tr_names = list(set(samples) - set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        if sr == True:
            self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}
        self.cnt_dict = {i: self.get_cnt(i) for i in names}
        # print(self.cnt_dict)
        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.label_dict={i: self.get_lbl(i) for i in names}
        self.mask_dict={i: self.get_img_label(i) for i in names}
        # self.exp_dict1 = {i:m[self.gene_set] for i,m in self.meta_dict.items()}
        # corrected = sc.external.pp.mnn_correct(self.exp_dict1,do_concatenate=False)
        # corrected =dict((y, x) for x, y in corrected)
        # print(corrected[0][0])
        # print(self.exp_dict1)
        # self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
        #                  dict(corrected[0][0]).items()}
        # print(corrected[0].shape)
        # print(self.exp_dict1)
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)

        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        centers_org = np.asarray(centers)
        # print(centers_org.shape)
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label_spot=self.label_dict[self.id2name[i]]
        img_mask=self.mask_dict[self.id2name[i]]
        # print(img_mask.shape)
        # print(centers[0].shape)
        if self.sr:
            im1 = self.img_dict1[self.id2name[i]]
            # im1 = np.swapaxes(im1,0,1)
            # print(im1.shape)
            # print(im.shape)
            cnt = cv2_detect_contour(im1, apertureSize=5, L2gradient=True)
            binary = np.zeros((im1.shape[0:2]), dtype=np.uint8)
            cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
            # Enlarged filter
            cnt_enlarged = scale_contour(cnt, 1.05)
            binary_enlarged = np.zeros(im1.shape[0:2])
            cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
            # img_new = im.copy()
            # cv2.drawContours(img_new, [cnt], -1, (255), thickness=50)
            # resize_factor = 1000 / np.min(im.shape[0:2])
            # resize_width = int(im.shape[1] * resize_factor)
            # resize_height = int(im.shape[0] * resize_factor)
            # img_new = cv2.resize(img_new, ((resize_width, resize_height)))
            # cv2.imwrite('./cnt.jpg', img_new)
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y
            index = 0
            index_list = []
            # print(binary_enlarged.shape,max_x,max_y)
            while y < max_y:
                x = min_x
                while x < max_x:
                    if binary_enlarged[y, x] != 0:
                        centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                        positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            # target_masks = torch.zeros((n_patches, 112,112,3))
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            # normalized_image = (image - min_val) / (max_val - min_val)
            for i in range(n_patches):
                # center = centers[i].cpu().numpy()
                # idx=np.where((centers_org[:,0] == center[0]) &(centers_org[:,1] == center[1]))
                # print(len(idx))
                # if i==0:
                #     print(center,centers_org)
                # center=centers_org[i]
                # if len(idx[0])>0:
                #     # result = np.where((centers_org[0] == center[0]) &(centers_org[1] == center[1]))
                #     # print(center,centers_org[result])
                #     index_list.append(i)
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                # patch = im_torch[(x - r):(x + r), (y - r):(y + r), :]

                # Normalize the patch
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # target_mask=img_mask[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                # Flatten and store the normalized patch
                patches[i, :] = normalized_patch.flatten()
                # target_masks[i]=target_mask
                # normalized_patch=(patch - min_val) / (max_val - min_val)
                # patches[i] = normalized_patch.flatten()
            # print(len(index_list))
            # index_list=np.asarray(index_list)
            # print(len(centers_org))
            # np.save('./index_list', index_list)
            # print(patches.shape)
            return patches, positions, centers,img_mask

        else:
            n_patches = len(centers)
            # print(len(centers_org))
            patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor

            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            # min_val = np.min(im)
            # max_val = np.max(im)
            # masks = torch.ones((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                normalized_patch = (patch - min_val) / (max_val - min_val)
                # target_mask = img_mask[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                # Flatten and store the normalized patch
                patches[i, :] = normalized_patch.flatten()
                # target_masks[i] = target_mask
                # patches[i] = patch.flatten()
            # print(patches.shape,label.shape ,positions.shape,exps.shape)
            if self.train:
                return patches, positions, exps,img_mask
            else:
                return patches, positions, exps, torch.Tensor(centers),img_mask

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name

        im = Image.open(path)
        return im

    def get_img1(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name

        im = cv2.imread(path)
        return im
    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1
            # final_labels = np.zeros_like(boundary_masks[0], dtype=np.int)
            # Visualization
            # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # ax[0].imshow(boundary_mask, cmap='gray')
            # ax[0].set_title('Original Boundary Mask')
            # ax[1].imshow(interior_mask, cmap='gray')
            # ax[1].set_title('Initial Interior Mask')
            # # plt.imshow(labels_im, cmap='nipy_spectral', interpolation='none')
            # # plt.title('Refined Labeled Image Excluding Inner Regions')
            # ax[2].imshow(labels_im, cmap='nipy_spectral', interpolation='none')
            # ax[2].set_title('Final Interior Mask Excluding Inner Circles')
            # plt.show()
            # # Invert the mask: so that the regions inside the boundaries are True
            # region_mask = np.invert(boundary_mask)
            # # print(np.unique(region_mask),'region')
            # # Assign a unique label to all regions within this color boundary
            # final_labels[region_mask] = idx
            # # # Label each connected component (region) with a unique integer
            # # labeled_regions, num_labels = label(region_mask, connectivity=2, return_num=True)
            # #
            # # # Update final labels: Add current labels multiplied by index to ensure uniqueness
            # # # Also, ensure that we do not overwrite labels from previous iterations
            # # labeled_regions[labeled_regions > 0] += max_label
            # #
            # # # Update the maximum label used
            # # min_val = np.min(labeled_regions)
            # # max_val = np.max(labeled_regions)
            # # print(f"For color {idx} ({boundary_color}): Min label = {min_val}, Max label = {max_val}")
            # #
            # # # Update final labels without overwriting existing labels
            # # mask_update = np.logical_and(final_labels == 0, labeled_regions > 0)
            # # final_labels[mask_update] = labeled_regions[mask_update] + (idx * 1000)
            # print(
            #     f"After updating with color {boundary_color}, final_labels min: {np.min(final_labels)}, max: {np.max(final_labels)}")
            # mask_update = np.logical_and(final_labels == 0, labeled_regions > 0)
            # final_labels[mask_update] = labeled_regions[mask_update] + idx * 1000  # Ensure unique labels
            # plt.show()
            # plt.imshow(boundary_mask)
            # plt.title("Boundary Mask for Color: " + str(boundary_color))
            # plt.show()
            # plt.imshow(region_mask)
            # plt.title("Region Mask for Color: " + str(boundary_color))
            # plt.show()
            # plt.savefig("Region Mask.png", bbox_inches='tight', pad_inches=0)
            # plt.close()
            # plt.imshow(labeled_regions, cmap="tab20b")
            # plt.title("Labeled Regions Directly for Color: " + str(boundary_color))
            # plt.colorbar()
            # plt.show()
        # # Visualization
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(im_array)
        # ax[0].set_title("Original Image")
        # ax[1].imshow(final_labels, cmap="tab20b")
        # ax[1].set_title("Final Labeled Regions")
        #
        # for a in ax:
        #     a.axis("off")
        #
        # plt.show()

        return torch.from_numpy(final_labels)


    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)

        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')

            x = df['x'].values
            y = df['y'].values
            x = np.around(x).astype(int)
            y = np.around(y).astype(int)
            id = []
            for i in range(len(x)):
                id.append(str(x[i]) + 'x' + str(y[i]))
            df['id'] = id
            df.drop('pixel_x', inplace=True, axis=1)
            df.drop('pixel_y', inplace=True, axis=1)
            df.drop('x', inplace=True, axis=1)
            df.drop('y', inplace=True, axis=1)
            label_tensor = self.convert_df_to_tensor(df)

            return label_tensor
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class ViT_LUNG_seg(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(ViT_LUNG_seg, self).__init__()

        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.lbl_img = 'data/her2st/data/ST-pat/img'
        self.r = 224 // 4
        self.label_encoder = LabelEncoder()  # Initialize label encoder
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.sr = sr

        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        # te_names=[]
        samples = names
        # print(names)
        # te_names = fold
        te_names = [samples[fold]]
        # print(te_names)
        tr_names = list(set(samples) - set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        if sr == True:
            self.img_dict1 = {i: self.get_img1(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}
        self.cnt_dict = {i: self.get_cnt(i) for i in names}
        # print(self.cnt_dict)
        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.label_dict={i: self.get_lbl(i) for i in names}
        self.mask_dict={i: self.get_img_label(i) for i in names}
        # Filter out items with empty masks
        self.names = [name for name in names if torch.any(self.mask_dict[name])]
        # print(self.names)
        # Update dictionaries to only include items with non-empty masks
        self.img_dict = {name: img for name, img in self.img_dict.items() if name in self.names}
        if sr:
            self.img_dict1 = {name: img for name, img in self.img_dict1.items() if name in self.names}
        self.meta_dict = {name: meta for name, meta in self.meta_dict.items() if name in self.names}
        self.cnt_dict = {name: cnt for name, cnt in self.cnt_dict.items() if name in self.names}
        self.exp_dict = {name: exp for name, exp in self.exp_dict.items() if name in self.names}
        self.label_dict = {name: lbl for name, lbl in self.label_dict.items() if name in self.names}
        self.mask_dict = {name: mask for name, mask in self.mask_dict.items() if name in self.names}
        # self.exp_dict1 = {i:m[self.gene_set] for i,m in self.meta_dict.items()}
        # corrected = sc.external.pp.mnn_correct(self.exp_dict1,do_concatenate=False)
        # corrected =dict((y, x) for x, y in corrected)
        # print(corrected[0][0])
        # print(self.exp_dict1)
        # self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
        #                  dict(corrected[0][0]).items()}
        # print(corrected[0].shape)
        # print(self.exp_dict1)
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)

        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        centers_org = np.asarray(centers)
        # print(centers_org.shape)
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label_spot=self.label_dict[self.id2name[i]]
        img_mask=self.mask_dict[self.id2name[i]]
        # print(centers[0].shape)
        if self.sr:
            patches, target_seg,centers=cut_into_patches_segmentation(im,img_mask,(112,112))
            # im1 = self.img_dict1[self.id2name[i]]
            # im1 = np.swapaxes(im1,0,1)
            # print(im1.shape)
            # print(im.shape)
            # cnt = cv2_detect_contour(im1, apertureSize=5, L2gradient=True)
            # binary = np.zeros((im1.shape[0:2]), dtype=np.uint8)
            # cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
            # # Enlarged filter
            # cnt_enlarged = scale_contour(cnt, 1.05)
            # binary_enlarged = np.zeros(im1.shape[0:2])
            # cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
            # # img_new = im.copy()
            # # cv2.drawContours(img_new, [cnt], -1, (255), thickness=50)
            # # resize_factor = 1000 / np.min(im.shape[0:2])
            # # resize_width = int(im.shape[1] * resize_factor)
            # # resize_height = int(im.shape[0] * resize_factor)
            # # img_new = cv2.resize(img_new, ((resize_width, resize_height)))
            # # cv2.imwrite('./cnt.jpg', img_new)
            # centers = torch.LongTensor(centers)
            # max_x = centers[:, 0].max().item()
            # max_y = centers[:, 1].max().item()
            # min_x = centers[:, 0].min().item()
            # min_y = centers[:, 1].min().item()
            # r_x = (max_x - min_x) // 30
            # r_y = (max_y - min_y) // 30
            #
            # centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            # positions = torch.LongTensor([0, 0]).view(1, -1)
            # x = min_x
            # y = min_y
            # index = 0
            # index_list = []
            # # print(binary_enlarged.shape,max_x,max_y)
            # while y < max_y:
            #     x = min_x
            #     while x < max_x:
            #         if binary_enlarged[y, x] != 0:
            #             centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
            #             positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
            #         x += 56
            #     y += 56
            #
            # centers = centers[1:, :]
            # positions = positions[1:, :]

            # n_patches = len(centers)
            # patches = torch.zeros((n_patches, patch_dim))
            # target_masks = torch.zeros((n_patches, 112, 112))
            # im_np = np.array(im)  # Convert the image object to a NumPy array
            # im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
            #
            # min_val = torch.min(im_torch)
            # max_val = torch.max(im_torch)
            # # normalized_image = (image - min_val) / (max_val - min_val)
            # for i in range(n_patches):
            #     # center = centers[i].cpu().numpy()
            #     # idx=np.where((centers_org[:,0] == center[0]) &(centers_org[:,1] == center[1]))
            #     # print(len(idx))
            #     # if i==0:
            #     #     print(center,centers_org)
            #     # center=centers_org[i]
            #     # if len(idx[0])>0:
            #     #     # result = np.where((centers_org[0] == center[0]) &(centers_org[1] == center[1]))
            #     #     # print(center,centers_org[result])
            #     #     index_list.append(i)
            #     center = centers[i]
            #     x, y = center
            #     patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
            #     # patch = im_torch[(x - r):(x + r), (y - r):(y + r), :]
            #
            #     # Normalize the patch
            #     normalized_patch = (patch - min_val) / (max_val - min_val)
            #
            #     target_mask = img_mask[(x - self.r):(x + self.r), (y - self.r):(y + self.r)]
            #     # Flatten and store the normalized patch
            #     patches[i, :] = normalized_patch.flatten()
            #     target_masks[i] = target_mask
                # patches[i] = patch.flatten()
                # normalized_patch=(patch - min_val) / (max_val - min_val)
                # patches[i] = normalized_patch.flatten()
            # print(len(index_list))
            # index_list=np.asarray(index_list)
            # print(len(centers_org))
            # np.save('./index_list', index_list)
            # print(patches.shape)
            return patches, positions, centers,target_masks,img.shape

        else:
            n_patches = len(centers)
            # print(len(centers_org))
            patches = torch.zeros((n_patches, patch_dim))
            target_masks = torch.zeros((n_patches, 112, 112))
            exps = torch.Tensor(exps)
            im_np = np.array(im)  # Convert the image object to a NumPy array
            im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
            print(im_torch.shape,img_mask.shape,'shape')
            min_val = torch.min(im_torch)
            max_val = torch.max(im_torch)
            # min_val = np.min(im)
            # max_val = np.max(im)
            # masks = torch.ones((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                # print(patch.shape)
                normalized_patch = (patch - min_val) / (max_val - min_val)
                target_mask = img_mask[(x - self.r):(x + self.r), (y - self.r):(y + self.r)]
                # Flatten and store the normalized patch
                patches[i, :] = normalized_patch.flatten()
                print(target_mask.shape)
                target_masks[i,:,:] = target_mask
                # patches[i] = patch.flatten()
            # print(patches.shape,label.shape ,positions.shape,exps.shape)
            if self.train:
                return patches, positions, exps,target_masks
            else:
                return patches, positions, exps, torch.Tensor(centers),target_masks

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name

        im = Image.open(path)
        return im

    def get_img1(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name

        im = cv2.imread(path)
        return im
    def get_img_label(self, name):
        path = self.lbl_img + '/' + name + '_annotated.png'
        if not os.path.exists(path):
            # print(f"No image found at {path}. Returning an empty tensor.")
            return torch.empty(0)
        im = Image.open(path)
        im_array = np.array(im.convert("RGB"))

        # Define the colors used for boundaries
        boundary_colors = [
                [255, 127, 39],  # Orange
                [14, 209, 69],  # Green
                [63, 72, 204],  # Blue
                [255, 242, 0],  # Yellow
                [112, 227, 223],  # Light Blue
                [236, 28, 36]  # Red
        ]

        # Initialize an empty array for final labeled regions
        final_labels = np.zeros(im_array.shape[:2], dtype=np.int)
        next_label = 1
        # Loop through each boundary color
        for idx, boundary_color in enumerate(boundary_colors, start=1):
            # Create a mask that identifies boundary pixels
            boundary_mask = np.all(im_array == boundary_color, axis=-1)
            boundary_mask = boundary_mask.astype(np.uint8)
            # print(np.unique(boundary_mask),'boundary')
            if not np.any(boundary_mask):
                # print(f"Boundary color {boundary_color} not found in the image. Skipping...")
                continue

            # Step 1: Ensure boundaries are closed
            kernel = np.ones((5, 5), np.uint8)
            closed_boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: Invert the image
            inverted_mask = ~closed_boundary_mask.astype(bool)

            # Step 3: Connected Component Labeling
            num_labels, labels_im = cv2.connectedComponents(inverted_mask.astype(np.uint8))

            # Step 4: Identify and Label Interior Regions
            # Assuming that label '1' corresponds to the exterior background
            interior_mask = np.where(labels_im > 1, 1, 0).astype(np.uint8)

            # Find contours and hierarchy
            contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and set values to 0 for contours inside another contour
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # If h[3] is not -1, the contour has a parent, i.e., it's inside another contour
                if h[3] != -1:
                    cv2.drawContours(interior_mask, [contour], -1, 0, thickness=-1)  # Set value to 0
            final_labels[interior_mask == 1] = next_label
            next_label += 1
            # final_labels = np.zeros_like(boundary_masks[0], dtype=np.int)
            # Visualization
            # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # ax[0].imshow(boundary_mask, cmap='gray')
            # ax[0].set_title('Original Boundary Mask')
            # ax[1].imshow(interior_mask, cmap='gray')
            # ax[1].set_title('Initial Interior Mask')
            # # plt.imshow(labels_im, cmap='nipy_spectral', interpolation='none')
            # # plt.title('Refined Labeled Image Excluding Inner Regions')
            # ax[2].imshow(labels_im, cmap='nipy_spectral', interpolation='none')
            # ax[2].set_title('Final Interior Mask Excluding Inner Circles')
            # plt.show()
            # # Invert the mask: so that the regions inside the boundaries are True
            # region_mask = np.invert(boundary_mask)
            # # print(np.unique(region_mask),'region')
            # # Assign a unique label to all regions within this color boundary
            # final_labels[region_mask] = idx
            # # # Label each connected component (region) with a unique integer
            # # labeled_regions, num_labels = label(region_mask, connectivity=2, return_num=True)
            # #
            # # # Update final labels: Add current labels multiplied by index to ensure uniqueness
            # # # Also, ensure that we do not overwrite labels from previous iterations
            # # labeled_regions[labeled_regions > 0] += max_label
            # #
            # # # Update the maximum label used
            # # min_val = np.min(labeled_regions)
            # # max_val = np.max(labeled_regions)
            # # print(f"For color {idx} ({boundary_color}): Min label = {min_val}, Max label = {max_val}")
            # #
            # # # Update final labels without overwriting existing labels
            # # mask_update = np.logical_and(final_labels == 0, labeled_regions > 0)
            # # final_labels[mask_update] = labeled_regions[mask_update] + (idx * 1000)
            # print(
            #     f"After updating with color {boundary_color}, final_labels min: {np.min(final_labels)}, max: {np.max(final_labels)}")
            # mask_update = np.logical_and(final_labels == 0, labeled_regions > 0)
            # final_labels[mask_update] = labeled_regions[mask_update] + idx * 1000  # Ensure unique labels
            # plt.show()
            # plt.imshow(boundary_mask)
            # plt.title("Boundary Mask for Color: " + str(boundary_color))
            # plt.show()
            # plt.imshow(region_mask)
            # plt.title("Region Mask for Color: " + str(boundary_color))
            # plt.show()
            # plt.savefig("Region Mask.png", bbox_inches='tight', pad_inches=0)
            # plt.close()
            # plt.imshow(labeled_regions, cmap="tab20b")
            # plt.title("Labeled Regions Directly for Color: " + str(boundary_color))
            # plt.colorbar()
            # plt.show()
        # # Visualization
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(im_array)
        # ax[0].set_title("Original Image")
        # ax[1].imshow(final_labels, cmap="tab20b")
        # ax[1].set_title("Final Labeled Regions")
        #
        # for a in ax:
        #     a.axis("off")
        #
        # plt.show()

        return torch.from_numpy(final_labels)


    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)

        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')

            x = df['x'].values
            y = df['y'].values
            x = np.around(x).astype(int)
            y = np.around(y).astype(int)
            id = []
            for i in range(len(x)):
                id.append(str(x[i]) + 'x' + str(y[i]))
            df['id'] = id
            df.drop('pixel_x', inplace=True, axis=1)
            df.drop('pixel_y', inplace=True, axis=1)
            df.drop('x', inplace=True, axis=1)
            df.drop('y', inplace=True, axis=1)
            label_tensor = self.convert_df_to_tensor(df)

            return label_tensor
        else:
            # print(f"Warning: Path {path} does not exist. Returning an empty tensor.")
            return torch.tensor([])

    def convert_df_to_tensor(self, df):
        # Extract relevant label information from DataFrame
        # This will depend on your specific use case
        relevant_label_info = df['label'].values

        # Convert string labels to integer labels
        integer_labels = self.label_encoder.fit_transform(relevant_label_info)

        # Convert to tensor
        label_tensor = torch.tensor(integer_labels, dtype=torch.long)

        return label_tensor

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        # print(cnt.shape)
        pos = self.get_pos(name)
        # print(pos.shape,'sfdsdafas')
        meta = cnt.join((pos.set_index('id')))
        # print(meta.shape)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,aug=False,norm=False,fold=0):
        super(ViT_SKIN, self).__init__()

        self.dir = '/ibex/scratch/pangm0a/spatial/data/GSE144240_RAW/'
        self.r = 224//4

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        test_names = ['P2_ST_rep2']

        # gene_list = list(np.load('data/skin_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))
        # gene_list = list(np.load('figures/mse_2000-vit_skin_a.npy',allow_pickle=True))

        self.gene_list = gene_list

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.ToTensor()
        ])
        self.norm = norm

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names
            # names = names[3:]
            # names = test_names
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = test_names
            names = te_names

        print('Loading imgs...')
        if self.aug:
            self.img_dict = {i: self.get_img(i) for i in names}
        else:
            self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))) for i,m in self.meta_dict.items()}
        else:
            self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        if self.aug:
            im = self.transforms(im)
            # im = im.permute(1,2,0)
            im = im.permute(2,1,0)
        else:
            im = im.permute(1,0,2)
            # im = im

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            n_patches = len(centers)
            patches = torch.zeros((n_patches,patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()


            return patches, positions, centers

        else:    
            n_patches = len(centers)
            
            patches = torch.zeros((n_patches,patch_dim))
            exps = torch.Tensor(exps)

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

            
                if self.train:
                    return patches, positions, exps
                else: 
                    return patches, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)


class SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(SKIN, self).__init__()

        self.dir = '/ibex/scratch/pangm0a/spatial/data/GSE144240_RAW/'
        self.r = 224//2

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        test_names = ['P2_ST_rep2']

        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list

        self.train = train
        self.sr = sr

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names
            # names = names[3:]
            # names = test_names
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = test_names
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i:self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x-self.r, y-self.r, x+self.r, y+self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else: 
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)


if __name__ == '__main__':
    # dataset = VitDataset(diameter=112,sr=True)
    dataset = ViT_HER2ST(train=True,mt=False)
    # dataset = ViT_SKIN(train=True,mt=False,sr=False,aug=False)

    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    # print(dataset[0][3].shape)
    # print(dataset.max_x)
    # print(dataset.max_y)
    # print(len(dataset.gene_set))
    # np.save('data/her_g_list.npy',dataset.gene_set)
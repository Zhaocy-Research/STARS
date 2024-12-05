import torch
from torch.utils.data import DataLoader
from utils import *
from vis_model import HisToGene
import warnings
from dataset import ViT_HER2ST, ViT_SKIN
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from vis_model import lung_finetune_flex,ReconstructionModel,SegmentationModel,BaseModel
warnings.filterwarnings('ignore')


MODEL_PATH = ''

# device = 'cpu'
def model_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            
            pred = model(patch, position)


            if preds is None:
                preds = pred.squeeze()
                ct = center
                gt = exp
            else:
                # print(pred.shape,preds.shape)
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)
                
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt

def sr_predict(model, test_loader, attention=True,device = torch.device('cpu')):
    model.phase = "segmentation"
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, center,mask in tqdm(test_loader):
            
            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)
            
            if preds is None:
                preds = pred.squeeze()
                ct = center
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct


    return adata


def stitch_patches_with_centers(patches, centers, output_shape, patch_h, patch_w):
    """
    Stitch patches back into an image using center coordinates.

    Parameters:
        patches (np.array): Patches of shape (num_patches, patch_h, patch_w, C).
        centers (np.array): Center coordinates for each patch of shape (num_patches, 2).
        output_shape (tuple): The desired output shape of the image (H, W, C).
        patch_h (int): The height of each patch.
        patch_w (int): The width of each patch.

    Returns:
        np.array: The reconstructed image of shape output_shape.
    """
    H, W,C = output_shape
    reconstructed_image = np.zeros((H, W))

    for idx, (center_y, center_x) in enumerate(centers):
        # Calculate the top-left corner of the patch
        start_y = int(center_y - patch_h // 2)
        start_x = int(center_x - patch_w // 2)

        # Ensure the patch is within the bounds of the output image
        if (
                start_y >= 0 and start_y + patch_h <= H and
                start_x >= 0 and start_x + patch_w <= W
        ):
            reconstructed_image[start_y:start_y + patch_h, start_x:start_x + patch_w, :] = patches[idx]

    return reconstructed_image
def sr_predict_lung_segmentation(model, test_loader, attention=True):
    # Specify the GPUs to use
    device_ids = [0, 2, 3]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    
    model.phase = "segmentation"
    model = model.to(device)
    
    # If multiple GPUs are available, wrap model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    model.eval()
    
    all_preds = None
    all_centers = None
    all_patches = None
    all_img_shapes = []
    
    with torch.no_grad():
        for patch, center, mask, img_shape in tqdm(test_loader):
            patch = patch.to(device)
            pred = model(patch)
            center = center.to(device)
            
            if all_patches is None and all_preds is None and all_centers is None:
                all_patches = patch.squeeze(dim=0)
                all_preds = pred.squeeze(dim=0)
                all_centers = center
            else:
                all_patches = torch.cat((all_patches, patch.squeeze(dim=0)), dim=0)
                all_preds = torch.cat((all_preds, pred.squeeze(dim=0)), dim=0)
                all_centers = torch.cat((all_centers, center), dim=0)
            
            all_img_shapes.append(img_shape)
        
    all_patches_np = all_patches.cpu().numpy()
    all_preds_np = all_preds.cpu().numpy()
    all_centers_np = all_centers.cpu().numpy()
    
    return all_patches_np, all_centers_np, all_preds_np, all_img_shapes
def sr_predict_lung_segmentation_spot(model, test_loader,device, attention=True):
    # Specify the GPUs to use
    #device_ids = [0, 2, 3]
    #device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    #model.phase = "segmentation"
    #model = model.to(device)
    
    # If multiple GPUs are available, wrap model with DataParallel
    #if torch.cuda.device_count() > 1:
        #print(f"Using {torch.cuda.device_count()} GPUs!")
        #model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    model.eval()
    
    all_preds = None
    all_centers = None
    all_patches = None
    all_img_shapes = []
    all_gene_maps=None
    all_patches_list = []
    all_preds_list = []
    all_centers_list = []
    all_gene_maps_list = []
    all_img_shapes_list = []
    with torch.no_grad():
        for patch, target,center, mask, img_shape in tqdm(test_loader):
            patch = patch.to(device)
            target = target.to(device)
            image_embeddings,spot_embeddings,loss = super(ReconstructionModel, model).forward(patch,target)
            #base_output=base_output.to('cuda:4')
            #gene_recon, gene_map, _=model.image_reconstruction_head(base_output)
            #gene_recon=gene_recon.to('cuda:3')
            image_embeddings = image_embeddings.to('cuda:1')
            spot_embeddings = spot_embeddings.to('cuda:1')

            # Concatenate the embeddings
            # Assuming you want to concatenate along the last dimension
            combined_embeddings = torch.cat([image_embeddings, spot_embeddings], dim=-1)

            # Pass the combined embeddings to the classification head
            gene_recon = model.classification_head(combined_embeddings)
            #gene_recon = model.classification_head(spot_embeddings)
            #pred,gene_map = model(patch)
            #center = center.to(device)
            center = center.to(device).squeeze(dim=0)
            #patch_cpu = patch.squeeze(dim=0).to('cpu')
            #gene_recon_cpu = gene_recon.squeeze(dim=0).to('cpu')
            #center_cpu = center.to('cpu')
            #gene_map_cpu = gene_map.squeeze(dim=0).to('cpu')
            # Initialize or concatenate the outputs on CPU
            #if all_patches is None:
            #    all_patches = patch_cpu
            #    all_centers = center_cpu
            #    all_gene_maps = gene_map_cpu
            #else:
            #    all_patches = torch.cat((all_patches, patch_cpu), dim=0)
            #    all_preds = torch.cat((all_preds, gene_recon_cpu), dim=0)
            #    all_centers = torch.cat([all_centers, center_cpu], dim=0)
            #    all_gene_maps = torch.cat((all_gene_maps, gene_map_cpu), dim=0)
            # Append the image shape to the list
            #all_img_shapes.append(img_shape)
            #This code snippet moves the tensors to the CPU before concatenating them. This will save GPU memory, but keep in mind that moving data between the CPU and GPU can be slow, so this might increase the overall time of your evaluation. If you're not facing out-of-memory errors on the GPU, it's usually faster to perform these operations on the GPU.





            #if all_patches is None and all_preds is None and all_centers is None and all_gene_maps is None:
            #    all_patches = patch.squeeze(dim=0)
            #    all_preds = gene_recon.squeeze(dim=0)
            #    all_centers = center
            #    all_gene_maps=gene_map.squeeze(dim=0)
            #else:
            #    all_patches = torch.cat((all_patches, patch.squeeze(dim=0)), dim=0)
            #    all_preds = torch.cat((all_preds, gene_recon.squeeze(dim=0)), dim=0)
            #    all_centers = torch.cat([all_centers, center], dim=0)
            #    all_gene_maps = torch.cat((all_gene_maps, gene_map.squeeze(dim=0)), dim=0)
            #all_img_shapes.append(img_shape)
            all_patches_list.append(patch.squeeze(dim=0).to('cpu'))
            all_preds_list.append(gene_recon.squeeze(dim=0).to('cpu'))
            all_centers_list.append(center.to('cpu'))
            #all_gene_maps_list.append(gene_map.squeeze(dim=0).to('cpu'))
            all_img_shapes_list.append(img_shape)

    # Concatenate lists along a new dimension
    #all_patches_np = torch.stack(all_patches_list, dim=0).numpy()
    #all_preds_np = torch.stack(all_preds_list, dim=0).numpy()
    #all_centers_np = torch.stack(all_centers_list, dim=0).numpy()
    #all_gene_maps_np = torch.stack(all_gene_maps_list, dim=0).numpy()
    #all_patches_np = all_patches.numpy()
    #all_preds_np = all_preds.numpy()
    #all_centers_np = all_centers.numpy()
    #all_gene_maps_np = all_gene_maps.numpy()
    all_patches_np = all_patches_list[0].numpy()
    all_preds_np=all_preds_list[0].numpy()
    all_centers_np=all_centers_list[0].numpy()
    #all_gene_maps_np=all_gene_maps_list[0].numpy()
    all_img_shapes=all_img_shapes_list[0]
    return all_patches_np, all_centers_np, all_preds_np,all_img_shapes
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for fold in [5,11,17,26]:
    for fold in range(12):
        # fold=30
        # tag = '-vit_skin_aug'
        # tag = '-cnn_her2st_785_32_cv'
        tag = '-vit_her2st_785_32_cv'
        # tag = '-cnn_skin_134_cv'
        # tag = '-vit_skin_134_cv'
        ds = 'HER2'
        # ds = 'Skin'

        print('Loading model ...')
        # model = STModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = VitModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = STModel.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt") 
        model = SpatialTransformer.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
        print('Loading data ...')

        # g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        g = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        # dataset = SKIN(train=False,ds=ds,fold=fold)
        dataset = ViT_HER2ST(train=False,mt=False,sr=True,fold=fold)
        # dataset = ViT_SKIN(train=False,mt=False,sr=False,fold=fold)
        # dataset = VitDataset(diameter=112,sr=True)

        test_loader = DataLoader(dataset, batch_size=16, num_workers=4)
        print('Making prediction ...')

        adata_pred, adata = model_predict(model, test_loader, attention=False)
        # adata_pred = sr_predict(model,test_loader,attention=True)

        adata_pred.var_names = g
        print('Saving files ...')
        adata_pred = comp_tsne_km(adata_pred,4)
        # adata_pred = comp_umap(adata_pred)
        print(fold)
        print(adata_pred)

        adata_pred.write('processed/test_pred_'+ds+'_'+str(fold)+tag+'.h5ad')
        # adata_pred.write('processed/test_pred_sr_'+ds+'_'+str(fold)+tag+'.h5ad')

        # quit()

if __name__ == '__main__':
    main()


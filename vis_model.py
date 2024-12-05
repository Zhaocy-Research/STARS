import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformer import ViT
import math
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules import ImageEncoder, ProjectionHead, ImageEncoder_ViT,ImageEncoder_ViT_large, ImageEncoder_ViT_L, ImageEncoder_CLIP, ImageEncoder_resnet101, ImageEncoder_resnet152
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, patch,target):
        # Getting Image and spot Features
        image_features = self.image_encoder(patch)
        spot_features = target
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)
        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return image_embeddings,spot_embeddings,loss.mean()
class Decoder(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
        # Example decoder architecture 
        self.decode = nn.Sequential( 
            nn.Linear(input_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, output_dim), 
            nn.Sigmoid()  # Assuming input is normalized to [0,1] 
        ) 
    def forward(self, x): 
        return self.decode(x)  
class FeedForward(nn.Module):
    def __init__(self, n_inp, n_out, activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x, indices=None):
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y

class ELU(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta
class CLIPModel_ViT(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT_large()
        #self.image_encoder = ImageEncoder_ViT()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = nn.Sequential(
                FeedForward(384, 256),
                FeedForward(256, 256),
                FeedForward(256, 256),
                FeedForward(256, 256))
        #ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature
        #self.decoder=FeedForward(384, spot_embedding, activation=ELU(alpha=0.01, beta=0.01))
        self.decoder=FeedForward(256, spot_embedding, activation=ELU(alpha=0.01, beta=0.01))
        #self.set_requires_grad(self.spot_projection, False)
    def set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad
    def forward(self, patch,target):
        # Getting Image and spot Features
        #print(patch.shape,'shape')
        image_features = self.image_encoder(patch)
        #image_features=image_features1[:,0,:]
        #spot_features = target
        image_features = image_features.to('cuda:1')
        #embs_mean = np.nanmean(image_features.cpu().detach().numpy(), axis=(0, 1))
        #embs_std = np.nanstd(image_features.cpu().detach().numpy(), axis=(0, 1))
        #image_features = (image_features - torch.tensor(embs_mean, device=image_features.device)) / (torch.tensor(embs_std, device=image_features.device) + 1e-12)

        #image_features1=image_features[:,0,:]
        #image_features = image_features.to('cuda:1')
        spot_features = target.to('cuda:1')
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings1 = self.image_projection(image_features)
        image_embeddings=image_embeddings1[:,0,:]
        #image_embeddings = image_features
        #print(image_features.shape,image_embeddings.shape)
        spot_embeddings = self.spot_projection(spot_features)
        #reconstructed_image=self.decoder(image_embeddings)
        # Calculating the Loss
        #mean = image_embeddings.mean(dim=1, keepdim=True)
        #std = image_embeddings.std(dim=1, keepdim=True, unbiased=False)
        #image_embeddings = (image_embeddings - mean) / (std + 1e-6)  # Prevent division by zero
        #mean = spot_embeddings.mean(dim=1, keepdim=True)
        #std = spot_embeddings.std(dim=1, keepdim=True, unbiased=False)
        #spot_embeddings = (spot_embeddings - mean) / (std + 1e-6)  # Prevent division by zero
        #image_embeddings=normalize_embeddings(image_embeddings)
        #spot_embeddings=normalize_embeddings(spot_embeddings)
        #image_embeddings=F.normalize(image_embeddings, p=2, dim=1)
        #spot_embeddings=F.normalize(spot_embeddings, p=2, dim=1)
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        #patch=patch.to('cuda:1')
        #reconstruction_loss=F.mse_loss(reconstructed_image,patch.view(patch.size(0),-1),reduction='mean')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        gene_predictions = self.decoder(image_embeddings1)
        gene_predictions_mean=gene_predictions.mean(-2)
        target=target.to('cuda:1')
        mse_loss=((gene_predictions_mean - target)**2).mean()
        #print(mse_loss.shape,loss.shape)
        total_loss=mse_loss+loss.mean()
        #total_loss=contrastive_loss.mean()+reconstruction_loss
        return gene_predictions,spot_features,total_loss
    #def normalize_embeddings(embeddings, dim=1):
    #    mean = embeddings.mean(dim=dim, keepdim=True)
    #    std = embeddings.std(dim=dim, keepdim=True, unbiased=False)
    #    normalized_embeddings = (embeddings - mean) / (std + 1e-6)  # Prevent division by zero
    #    return normalized_embeddings
class CLIPModel_ViT1(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT_large()
        #self.image_encoder = ImageEncoder_ViT()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature
        self.decoder=FeedForward(256, spot_embedding, activation=ELU(alpha=0.01, beta=0.01))
        #self.set_requires_grad(self.image_encoder, False)
        #self.set_requires_grad(self.image_projection,True)
        #self.set_requires_grad(self.spot_projection, False)

        # Ensure requires_grad is True for the decoder
        #self.set_requires_grad(self.decoder, True)
    def set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad
    def forward(self, patch,target):
        # Getting Image and spot Features
        #print(patch.shape,'shape')
        image_features = self.image_encoder(patch)
        #image_features=image_features1[:,0,:]
        spot_features = target
        image_features = image_features.to('cuda:1')
        #embs_mean = np.nanmean(image_features.cpu().detach().numpy(), axis=(0, 1))
        #embs_std = np.nanstd(image_features.cpu().detach().numpy(), axis=(0, 1))
        #image_features = (image_features - torch.tensor(embs_mean, device=image_features.device)) / (torch.tensor(embs_std, device=image_features.device) + 1e-12)

        #image_features1=image_features[:,0,:]
        #image_features = image_features.to('cuda:1')
        spot_features = target.to('cuda:1')
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings1 = self.image_projection(image_features)
        image_embeddings=image_embeddings1[:,0,:]
        #image_embeddings = image_features
        #print(image_features.shape,image_embeddings.shape)
        spot_embeddings = self.spot_projection(spot_features)
        #reconstructed_image=self.decoder(image_embeddings)
        # Calculating the Loss
        #mean = image_embeddings.mean(dim=1, keepdim=True)
        #std = image_embeddings.std(dim=1, keepdim=True, unbiased=False)
        #image_embeddings = (image_embeddings - mean) / (std + 1e-6)  # Prevent division by zero
        embs_mean = torch.mean(image_embeddings1, dim=(0, 1),keepdim=True)
        embs_std = torch.std(image_embeddings1, dim=(0, 1), keepdim=True)
        image_embeddings1 = (image_embeddings1 - embs_mean) / (embs_std+ 1e-6)
        #image_embeddings1 -= embs_mean
        #image_embeddings1 /= embs_std + 1e-12
        #mean = spot_embeddings.mean(dim=1, keepdim=True)
        #std = spot_embeddings.std(dim=1, keepdim=True, unbiased=False)
        #spot_embeddings = (spot_embeddings - mean) / (std + 1e-6)  # Prevent division by zero
        #image_embeddings=normalize_embeddings(image_embeddings)
        #spot_embeddings=normalize_embeddings(spot_embeddings)
        #image_embeddings=F.normalize(image_embeddings, p=2, dim=1)
        #spot_embeddings=F.normalize(spot_embeddings, p=2, dim=1)
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        #patch=patch.to('cuda:1')
        #reconstruction_loss=F.mse_loss(reconstructed_image,patch.view(patch.size(0),-1),reduction='mean')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        gene_predictions = self.decoder(image_embeddings1[:,1:,:])
        gene_predictions_mean=gene_predictions.sum(-2)
        target=target.to('cuda:1')
        mse_loss=((gene_predictions_mean - target)**2).mean(dim=-1)
        #print(mse_loss.shape,loss.shape)
        total_loss=mse_loss+loss
        #total_loss=contrastive_loss.mean()+reconstruction_loss
        return gene_predictions,spot_embeddings,total_loss.mean()
    #def normalize_embeddings(embeddings, dim=1):
    #    mean = embeddings.mean(dim=dim, keepdim=True)
    #    std = embeddings.std(dim=dim, keepdim=True, unbiased=False)
    #    normalized_embeddings = (embeddings - mean) / (std + 1e-6)  # Prevent division by zero
    #    return normalized_embeddings

class CLIPModel_CLIP(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_CLIP()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, patch,target):
        # Getting Image and spot Features
        image_features = self.image_encoder(patch)
        spot_features = target
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return image_embeddings,spot_embeddings,loss.mean()

class CLIPModel_ViT_L(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT_L()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, patch,target):
        # Getting Image and spot Features
        image_features = self.image_encoder(patch)
        spot_features = target
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return image_embeddings,spot_embeddings,loss.mean()


class CLIPModel_resnet101(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet101()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, patch,target):
        # Getting Image and spot Features
        image_features = self.image_encoder(patch)
        spot_features = target
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return image_embeddings,spot_embeddings,loss.mean()

class CLIPModel_resnet152(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet152()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, patch,target):
        # Getting Image and spot Features
        image_features = self.image_encoder(patch)
        spot_features = target
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return image_embeddings,spot_embeddings,loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class PatchEmbeddingCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024, patch_size=112):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling (GAP) layer
        
    def forward(self, x):
        # Assuming x is (B, N, P, P, 3)
        B, N, P, _,C = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(-1, C, P, P)  # Reshape to (B*N, 3, P, P)
        x_transformer = torch.zeros(B, N, 1024, device=x.device)
        chunk_size=512
        x_transformer_list = []

        for i in range(0, B*N, chunk_size):
            x_chunk = x[i:i+chunk_size]  # Select chunk of patches
    
            # Convolution operations
            x1 = F.relu(self.conv1(x_chunk))  
            x2 = F.relu(self.conv2(x1))       
            x3 = F.relu(self.conv3(x2))       
    
            # Global Average Pooling
            x_chunk_transformer = self.global_avg_pool(x3).squeeze(-1).squeeze(-1)  # [chunk_size, C']
    
            x_transformer_list.append(x_chunk_transformer)

        # Concatenate along the batch dimension
        x_transformer = torch.cat(x_transformer_list, dim=0).reshape(B, N, -1)
        # Convolution operations
        #x1 = F.relu(self.conv1(x))  # Intermediate spatial features 1
        #x2 = F.relu(self.conv2(x1)) # Intermediate spatial features 2
        #x3 = F.relu(self.conv3(x2)) # Intermediate spatial features 3
        
        # Global Average Pooling
        #x_transformer = self.global_avg_pool(x3)
        
        # Reshape back to (B, N, dim)
        #_, C, _, _ = x_transformer.shape
        #x_transformer = x_transformer.reshape(B, N, C)
        
        # Choose which spatial features to return for reconstruction
        #spatial_features = x2  # or x1, x3 depending on your use case
        
        return x_transformer
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # Adjust the out_channels of value_conv to match query and key
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width*height)
        key = self.key_conv(x).view(batch_size, -1, width*height)
        #print("Input x dimensions:", x.size())

        attention = self.softmax(torch.bmm(query, key.transpose(1, 2)))  # Transpose key tensor
        value = self.value_conv(x).view(batch_size, -1, width*height)
        #print("Query shape:", query.shape)
        #print("Key shape:", key.shape)
        #print("Attention shape:", attention.shape)
        #print("Value shape:", value.shape)

        out = torch.bmm(attention, value)
        return out.mean(dim=-1)

class ModifiedImageDecoder(nn.Module):
    def __init__(self, input_dim, n_genes, output_channels=3, small_dim=64, dropout_rate=0.5):
        super(ModifiedImageDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, 512*7*7)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.n_genes=n_genes
        # Layer to produce smaller gene expression map
        self.deconv4_small_gene = nn.ConvTranspose2d(64, small_dim, kernel_size=4, stride=2, padding=1)
        self.small_dim=small_dim
        # Attention pooling and mapping to n_genes
        self.attention_pool = AttentionPooling(small_dim)
        self.fc1 = nn.Linear(small_dim//8, small_dim//2)
        self.fc2 = nn.Linear(small_dim//2, small_dim)
        self.fc3 = nn.Linear(small_dim, small_dim*2)
        
        # Final layer mapping to n_genes
        self.fc_gene = nn.Linear(small_dim*2, n_genes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        #self.fc_gene = nn.Linear(small_dim//8, n_genes)
        
        # Layer to produce image patch
        self.deconv4_image = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0)*x.size(1), 512, 7, 7)
        n = x.size(0)
        center_gene_expressions = torch.empty((n, self.n_genes), device=x.device)
        gene_maps = torch.empty((n, self.small_dim, 112, 112), device=x.device)  # Replace with appropriate values
        image_patches = torch.empty((n, 3, 112, 112), device=x.device)  # Replace with appropriate values
    
        #center_gene_expressions = []
        #gene_maps = []
        #image_patches = []
        batch_size=64
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mini_batch = x[start:end]
            
            mb = self.relu(self.deconv1(mini_batch))
            mb = self.relu(self.deconv2(mb))
            mb = self.relu(self.deconv3(mb))
            
            # Smaller gene expression map
            small_gene_map = self.deconv4_small_gene(mb)
            
            # Use attention mechanism to compute center gene expression
            center_gene_expression = self.attention_pool(small_gene_map).squeeze(-1).squeeze(-1)
            center_gene_expression = self.dropout(self.relu(self.fc1(center_gene_expression)))
            center_gene_expression = self.dropout(self.relu(self.fc2(center_gene_expression)))
            center_gene_expression = self.fc_gene(self.relu(self.fc3(center_gene_expression)))
            #center_gene_expression = self.fc_gene(center_gene_expression)
            
            # Image patch
            image_patch = self.relu(self.deconv4_image(mb))
            image_patch = torch.clamp(image_patch, 0, 255)  # Ensuring pixel values are within [0, 255]
            center_gene_expressions[start:end] = center_gene_expression
            gene_maps[start:end] = small_gene_map
            image_patches[start:end] = image_patch
            #center_gene_expressions.append(center_gene_expression)
            #gene_maps.append(small_gene_map)
            #image_patches.append(image_patch)
        
        return center_gene_expressions, gene_maps, image_patches
class ImageReconstructionHead(nn.Module):
    def __init__(self, vit_output_dim, spatial_feature_dim, output_channels):
        super().__init__()
        
        # Define a layer to process the ViT output
        self.vit_processing = nn.Linear(vit_output_dim, spatial_feature_dim * 4 * 4)  # Adjust the multiplier as per need
        
        # Define a decoder block (you might need more of these, depending on your use case)
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(spatial_feature_dim, spatial_feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(spatial_feature_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final layer to reconstruct the image
        self.final_layer = nn.Conv2d(spatial_feature_dim // 2, output_channels, kernel_size=3, padding=1)
        
    def forward(self, vit_output, spatial_features):
        # vit_output: [B, vit_dim]
        # spatial_features: [B, spatial_dim, H', W']
        
        # Process the ViT output and reshape it to have spatial dimensions
        vit_output = self.vit_processing(vit_output)
        vit_output = vit_output.view(vit_output.size(0), -1, 4, 4)  # Adjust the shape as per need
        
        # Concatenate the processed ViT output with the spatial features along the channel dimension
        # Ensure the spatial dimensions (H', W') are the same for concatenation
        spatial_features_up = F.interpolate(spatial_features, size=(4, 4), mode='bilinear', align_corners=False)
        x = torch.cat([vit_output, spatial_features_up], dim=1)
        
        # Pass through the decoder block(s)
        x = self.decoder_block(x)
        
        # Reconstruct the image
        reconstructed_image = self.final_layer(x)
        
        return reconstructed_image
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(3),
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class ImageDecoder(nn.Module):
    def __init__(self, input_dim, output_channels):
        super(ImageDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, 512*8*8)  
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0)*x.size(1), 512, 8, 8)  
        x = self.relu(self.deconv1(x))  
        x = self.relu(self.deconv2(x))  
        x = self.relu(self.deconv3(x))  
        x = self.relu(self.deconv4(x))  # Using ReLU as the final activation
        x = torch.clamp(x, 0, 255)  # Ensuring pixel values are within [0, 255]
        return x
class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet101'):
        super(FeatureExtractor, self).__init__()
        # backbone = torchvision.models.resnet101(pretrained=True)
        backbone = torchvision.models.squeezenet1_1(pretrained=True)
        # print(backbone)
        # patch_dim = 512 * 13 * 13
        # dim = 1024
        # self.patch_embedding = nn.Linear(patch_dim, dim)
        layers = list(backbone.children())[:-1]
        # print(layers)
        self.backbone = nn.Sequential(*layers)
        # self.backbone = backbone
    def forward(self, x):
        # print(x.shape)
        x = self.backbone(x)
        x=x.mean([2,3])
        # print(x.shape)
        # x=x.view(x.shape[0],-1)
        # x=self.patch_embedding(x)
        # print(x.shape)
        return x
class FeatureExtractor1(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet101'):
        super(FeatureExtractor1, self).__init__()
        patch_size = 112
        # patch_dim = 3 * patch_size * patch_size
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        # self.down3 = Down(128, 256)
        # self.down4 = Down(256, 256)
        # factor = 2
        # self.down5 = Down(256, 512 // factor)

        # dim=1024
        # self.patch_embedding = nn.Linear(patch_dim, dim)
        # self.backbone=nn.Sequential(self.patch_embedding)
        # backbone = torchvision.models.resnet101(pretrained=True)
        # backbone = torchvision.models.mobilenet_v3_small(pretrained=False)
        # # print(backbone)
        # layers = list(backbone.children())[:-1]
        # # print(layers)
        # self.backbone = nn.Sequential(*layers)
        # self.backbone = backbone
    def forward(self, x):
        x=self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        # x = self.down3(x)
        # x = self.down4(x)
        # x = self.down5(x)
        # x = x.mean([2, 3])
        # print(x.shape)
        x=x.view(x.shape[0],-1)
        # x = self.backbone(x)
        return x

class ViTDecoder(nn.Module):
    def __init__(self):
        super(ViTDecoder, self).__init__()

        # Custom Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),  # First layer takes input of size 256
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 224 * 224 * 3)  # Assuming the original image size is 224x224x3
        )

    def forward(self, encoded_features):
        # Decode
        x_reconstructed = self.decoder(encoded_features)
        x_reconstructed = x_reconstructed.view(-1, 3, 224, 224)  # Reshape to image dimensions

        return x_reconstructed
class BaseModel(nn.Module):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=128):
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate
        patch_dim = 3 * patch_size * patch_size
        num_classes=3
        #self.patch_embedding = PatchEmbeddingCNN(in_channels=3, out_channels=dim)
        self.dim=1024
        self.vit = CLIPModel_ViT(temperature=1.0,image_embedding=384,spot_embedding=1000)
        #self.patch_embedding = nn.Conv2d(1, dim, kernel_size=16, stride=16)
        #self.vit = nn.Transformer(dim, num_encoder_layers=12)
        #self.dim = dim

    def forward(self, patches,target):
        device = patches.device
        B, N, P, P, C = patches.size()
        B, N, C1 = target.size()
        patches= patches.view(B * N, C, P, P)

        # Reshape target to [B*N, C1]
        target = target.view(B * N, C1)
        patches=patches.to(device)
        target=target.to(device)
        #x_transformer = self.patch_embedding(patches)
        #x_transformer=x_transformer.to('cuda:1')
        #pos_enc = self.positional_encoding(N, x_transformer.device)
        #x_flat = x_transformer+ pos_enc
        image_embeddings,spot_embeddings,loss= self.vit(patches,target)
        #vit_output=vit_output.to('cuda:2')
        #x = self.patch_embedding(x)
        #x = x.flatten(2).permute(2, 0, 1)
        #x = self.vit(x)
        return image_embeddings,spot_embeddings,loss

    #def on_train_start(self):
    #    self.patch_embedding.to('cuda:0')
    #    self.vit.to('cuda:1')

    #def on_validation_start(self):
    #    self._move_components_to_devices()

    def _move_components_to_devices(self):
        self.patch_embedding.to('cuda:0')
        self.vit.to('cuda:1')

    def positional_encoding(self, seq_len, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim))
        pos_enc = torch.zeros((1, seq_len, self.dim), device=device)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        return pos_enc

class ReconstructionModel(BaseModel):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=128):
        super(ReconstructionModel, self).__init__(patch_size, n_layers, n_genes, dim, learning_rate, dropout, n_pos)
        #self.image_reconstruction_head = ViTDecoder()
        #gpu_ids = [2, 3,4,5,6,7]
        #if torch.cuda.device_count() > 1:
            #self.image_reconstruction_head = nn.DataParallel(self.image_reconstruction_head, device_ids=gpu_ids)
        #self.learning_rate = learning_rate
    def forward(self, patches,target):
        image_embeddings,spot_embeddings,loss = super().forward(patches,target)
        #image_embeddings=image_embeddings.to('cuda:2')
       # x_reconstructed = self.image_reconstruction_head(image_embeddings)
        #return image_embeddings,spot_embeddings,loss,x_reconstructed
        return image_embeddings,spot_embeddings,loss

    def _move_components_to_devices(self):
        super()._move_components_to_devices()
        self.image_reconstruction_head.to('cuda:2')
class HisToGene(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=64):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        patch_dim = 3*patch_size*patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim, dropout = dropout, emb_dropout = dropout)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        # print(patches.shape)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        return x

    def training_step(self, batch, batch_idx):        
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch

        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp = batch
        # print(patch.shape,exp.shape)
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
class lung_finetune_flex(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=128):
        super().__init__()
        self.learning_rate = learning_rate
        patch_dim = 3 * patch_size * patch_size
        num_classes=3
        #self.computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.patch_embedding = PatchEmbeddingCNN(in_channels=3, out_channels=dim)
        self.dim=1024
        # self.x_embed = nn.Embedding(n_pos, dim)
        # self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2 * dim, dropout=dropout, emb_dropout=dropout)
        #print(next(self.vit.parameters()).device,"first")

        #print(next(self.vit.parameters()).device)

        vit_output_dim = dim  # Assuming the ViT output dimension is equal to `dim`
        spatial_feature_dim = patch_size * patch_size  # Speculative, adjust as needed
        output_channels = 3
        self.phase = "reconstruction"  # Set initial phase
        #self.gene_head = nn.Sequential(
        #    nn.LayerNorm(dim),
        #    nn.Linear(dim, n_genes)
        #)
        self.image_reconstruction_head = IModifiedImageDecoder(input_dim=dim, n_genes=n_genes)
        # self.segmentation_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, 1)  # Assume binary segmentation, adjust if needed
        # )

        # Image reconstruction head
        #self.image_reconstruction_head = nn.Sequential(
        #    nn.LayerNorm(dim),
        #    nn.Linear(dim, patch_dim),
        #    nn.Sigmoid()  # Ensure pixel values are in [0, 1]
       # )
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
        # self.segmentation_head = nn.Sequential(
        #     nn.Linear(dim, patch_dim),
        #     nn.ReLU(),
        # )
        # self.conv=nn.Conv2d(3, 6, kernel_size=3, padding=1)
        #self.register_buffer("pos_enc", self.positional_encoding(seq_len))
    def on_train_start(self):
        # Assign devices to sub-models
        self.patch_embedding.to('cuda:0')
        self.vit.to('cuda:1')
        self.gene_head.to('cuda:2')
        self.image_reconstruction_head.to('cuda:2')
        self.classification_head.to('cuda:2')
    def on_validation_start(self):
        self._move_components_to_devices()

    #def on_test_start(self):
     #   self._move_components_to_devices()

    def _move_components_to_devices(self):
        # Assign devices to sub-models
        self.patch_embedding.to('cuda:0')
        self.vit.to('cuda:1')
        #self.gene_head.to('cuda:2')
        self.image_reconstruction_head.to('cuda:2')
        self.classification_head.to('cuda:2')
    def positional_encoding(self, seq_len, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim))
        pos_enc = torch.zeros((1, seq_len, self.dim), device=device)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, patches):
        #_, seq_len, _ = patches.size()
        device = patches.device
        B, N, P, P, C = patches.size()
        patches=patches.to('cuda:0')
        # Permute patches to [B*N, C, P, P]
        #patches = patches.permute(0, 1, 4, 2, 3).reshape(B*N, C, P, P)
    
        # Embedding patches using a convolutional layer
        x_transformer = self.patch_embedding(patches)
        x_transformer=x_transformer.to('cuda:1')
        #D = x_transformer.size(-1)  # Embedding dimension
        #x_flat = x_transformer.view(B, N, D)
        # Reshape to [B, N, D]
        #D = patch_embeddings.size(1)  # Embedding dimension
        #x_flat = patch_embeddings.flatten(2).transpose(1, 2)  # [B, N, dim], where N = H' * W'
        pos_enc = self.positional_encoding(N, x_transformer.device)
        x_flat = x_transformer+ pos_enc
        #print(f"Device for x_flat: {x_flat.device}")
        #print(next(self.vit.parameters()).device)  # Should show 'cuda:1'
        #print(next(self.vit.transformer.parameters()).device)  # Should also show 'cuda:1'

        # Passing through Vision Transformer
        vit_output = self.vit(x_flat)
        vit_output=vit_output.to('cuda:2')
        #pos_enc = self.positional_encoding(N,device)
        #x = patch_embeddings + pos_enc
        #pos_enc = self.positional_encoding(seq_len, device)
        #pos_enc = self.positional_encoding(seq_len)
        #patches = self.patch_embedding(patches)
        # print("Centers shape: ", centers.shape)
        # print("Centers dtype: ", centers.dtype)
        # print("Centers device: ", centers.device)
        # max_index_x = centers[:, :, 0].max().item()
        # max_index_y = centers[:, :, 1].max().item()
        #
        # assert max_index_x < 128, f"Maximum index in centers_x ({max_index_x}) is out of range of embedding size (128)"
        # assert max_index_y < 128, f"Maximum index in centers_y ({max_index_y}) is out of range of embedding size (128)"
        #
        # centers_x = self.x_embed(centers[:, :, 0])
        # centers_y = self.y_embed(centers[:, :, 1])
        # print(patches.shape,pos_enc.shape,'out')
        #x = patches + pos_enc
        #h = self.vit(x)
        # print(h.shape,'shape')
        if self.phase == "reconstruction":
            #gene_recon = self.gene_head(vit_output )
            gene_recon, gene_map, image_recon = self.image_reconstruction_head(vit_output)
            return gene_recon, image_recon
        elif self.phase == "segmentation":
            x = self.classification_head(vit_output)
            # x=x.view(-1, h.shape(1),112,112,3)
            # x = x.permute(0, 4, 1, 2, 3)

            return x
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

    def one_hot_encode(self,labels, num_classes):
        return torch.eye(num_classes)[labels]
    def check_for_invalid_values(self,tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values!")
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values!")


    def training_step(self, batch, batch_idx):
        patch, target_gene, target_seg = batch
        if self.phase == "reconstruction":
            pred_gene, pred_image = self(patch)
            pred_gene=pred_gene.to('cuda:0')
            pred_image=pred_image.to('cuda:0')
            # print(pred_gene.device,target_gene.device,'gene')
            # target_gene_cpu = target_gene.cpu()
            #
            # self.check_for_invalid_values(target_gene_cpu, "target_gene")
            # pred_gene_cpu = pred_gene.cpu()
            # self.check_for_invalid_values(pred_gene_cpu, "pred_gene")
            # print("pred_gene device: ", pred_gene.device)
            # print("target_gene device: ", target_gene.device)
            # print("pred_gene dtype: ", pred_gene.dtype)
            # print("target_gene dtype: ", target_gene.dtype)
            # if pred_gene.nelement() != target_gene.nelement():
            #     print("Warning: pred_gene and target_gene have different numbers of elements!")
            # try:
            #     loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            # except Exception as e:
            #     print("Error during loss calculation: ", str(e))

            # Check the range of target images
            # print(patch.min(), patch.max(),'min_max value')

            # Check the shapes of predictions and target
            # print(pred_image.shape, patch.shape)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            # print(loss_gene)
            loss_image = F.mse_loss(pred_image.view_as(patch), patch)
            # print(loss_image)
            loss = loss_gene + loss_image
            self.log('loss', loss, on_epoch=True, prog_bar=True)
            self.log('train_loss_recon', loss, on_epoch=True)
        elif self.phase == "segmentation":
            pred_seg = self(patch)

            pred_seg=pred_seg.to('cuda:0')
            # print("pred_seg min value:", torch.min(pred_seg))
            # print("pred_seg max value:", torch.max(pred_seg))
            # print("Any NaN values in pred_seg:", torch.isnan(pred_seg).any())
            # print("Any Inf values in pred_seg:", torch.isinf(pred_seg).any())
            # print("Example logits for the first instance:", pred_seg[0, 0, :])
            # softmaxed_pred_seg = torch.softmax(pred_seg, dim=-1)
            # print("Example softmaxed logits for the first instance:", softmaxed_pred_seg[0, 0, :])

            # labels_one_hot = self.one_hot_encode(target_seg, num_classes=3).to(device=pred_seg.device, dtype=pred_seg.dtype)
            # if labels_one_hot.shape[-1] > 1:  # assuming labels are one-hot encoded
            #     target_seg = torch.argmax(labels_one_hot, dim=-1)
            target_seg = target_seg.to(device=pred_seg.device, dtype=torch.long)  # ensure the target is long type
            pred_seg = pred_seg.view(-1, 3)  # [B*N, C]
            target_seg = target_seg.view(-1)  # [B*N]
            # assert pred_seg.device == target_seg.device, "Device mismatch"
            # assert target_seg.dtype == torch.long, "Incorrect dtype for target_seg"
            # assert pred_seg.dtype == torch.float32, "Incorrect dtype for pred_seg"
            # assert not torch.isnan(target_seg).any(), "NaN values in target_seg"
            # assert not torch.isinf(target_seg).any(), "Inf values in target_seg"
            # print(pred_seg)
            # print("target_seg unique values:", torch.unique(target_seg))
            # print("target_seg type:", target_seg.dtype)

            # Move target to the same device as predictions
            # target_seg = target_seg.to(device=pred_seg.device, dtype=torch.long)  # ensure the target is long type
            # print(target_seg.shape,pred_seg.shape,'shape')
            # print(labels_one_hot.shape)
            # labels_one_hot = labels_one_hot.view(pred_seg.shape[1], 4)
            criterion = nn.CrossEntropyLoss()
            loss=criterion(pred_seg, target_seg)
            # loss = F.cross_entropy(pred_seg, target_seg.long())
            self.log('loss', loss, on_epoch=True, prog_bar=True)
            self.log('train_loss_seg', loss,on_epoch=True)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def validation_step(self, batch, batch_idx):
        patch,target_gene, target_seg = batch  # assuming masks are the segmentation ground truth

        if self.phase == "reconstruction":
            pred_gene, pred_image = self(patch)
            pred_gene=pred_gene.to('cuda:0')
            pred_image=pred_image.to('cuda:0')
            # Check the range of target images
            # print(patch.min(), patch.max(),'min_max value')

            # Check the shapes of predictions and target
            # print(pred_image.shape, patch.shape)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss_image = F.mse_loss(pred_image.view_as(patch), patch)
            loss = loss_gene + loss_image
            self.log('val_loss', loss, on_epoch=True, prog_bar=True)
            self.log('val_loss_recon', loss, on_epoch=True)
            #self.log('loss', loss,  prog_bar=True)
            #self.log('eval_loss_recon', loss)
        elif self.phase == "segmentation":
            pred_seg = self(patch)
            pred_seg=pred_seg.to('cuda:0')
            # labels_one_hot = self.one_hot_encode(target_seg, num_classes=3).to(device=pred_seg.device,
            #                                                                    dtype=pred_seg.dtype)
            # labels_one_hot = labels_one_hot.view(pred_seg.shape(0), 4)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred_seg, target_seg)
            # loss = F.cross_entropy(pred_seg, target_seg.long())
            self.log('val_loss', loss, on_epoch=True, prog_bar=True)
            self.log('val_loss_seg', loss, on_epoch=True)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")
        # # Logging
        # self.log('val_loss', total_loss, prog_bar=True)
        # self.log('val_recon_loss', recon_loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        patch, target_gene, target_seg = batch  # assuming masks are the segmentation ground truth
        
        # Extract patches and their spatial locations from images
        # patches, locations = self.extract_patches(images)

        if self.phase == "reconstruction":
            pred_gene, pred_image = self(patch)
            pred_gene=pred_gene.to('cuda:0')
            pred_image=pred_image.to('cuda:0')
            # Check the range of target images
            # print(patch.min(), patch.max(),'min_max value')

            # Check the shapes of predictions and target
            # print(pred_image.shape, patch.shape)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss_image = F.mse_loss(pred_image.view_as(patch), patch)
            loss = loss_gene + loss_image
            self.log('loss', loss,  prog_bar=True)
            self.log('test_loss_recon', loss)
        elif self.phase == "segmentation":
            pred_seg = self(patch)
            pred_seg=pred_seg.to('cuda:0')
            # labels_one_hot = self.one_hot_encode(target_seg, num_classes=3).to(device=pred_seg.device,
            #                                                                    dtype=pred_seg.dtype)
            # labels_one_hot = labels_one_hot.view(pred_seg.shape(0), 4)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred_seg, target_seg)
            # loss = F.cross_entropy(pred_seg, target_seg.long())
            self.log('loss', loss,  prog_bar=True)
            self.log('test_loss_seg', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        # Optionally, you might want to save the predictions for further analysis
        # self.save_predictions(images, seg_maps, gene_preds, genes)

        return loss
    def reconstruction_parameters(self):
        return list(self.gene_head.parameters()) + list(self.image_reconstruction_head.parameters())
    def print_device_for_submodels(self):
        for name, submodule in self.named_children():
            print(f"Device for {name}: {next(submodule.parameters()).device}")
    def segmentation_parameters(self):
        return list(self.classification_head.parameters())
    def configure_optimizers(self):
        if self.phase == "reconstruction":
            optimizer = torch.optim.Adam(self.reconstruction_parameters(), lr=1e-3)
        elif self.phase == "segmentation":
            optimizer = torch.optim.Adam(self.segmentation_parameters(), lr=1e-5)
        return optimizer    
class SegmentationModel(ReconstructionModel):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=128, num_classes=3):
        super(SegmentationModel, self).__init__(patch_size, n_layers, n_genes, dim, learning_rate, dropout, n_pos)
        self.classification_head = nn.Sequential(
            nn.Linear(n_genes, num_classes)
        )
        #self.learning_rate = learning_rate

    def forward(self, patches,target):
        #image_embeddings,spot_embeddings,loss1,x_reconstructed= super().forward(x)
        image_embeddings,spot_embeddings,loss1= super().forward(patches,target)
        image_embeddings = image_embeddings.to('cuda:1')
        spot_embeddings = spot_embeddings.to('cuda:1')

        # Concatenate the embeddings
        # Assuming you want to concatenate along the last dimension
        combined_embeddings = torch.cat([image_embeddings, spot_embeddings], dim=-1)

        # Pass the combined embeddings to the classification head
        gene_recon = self.classification_head(combined_embeddings)
        return image_embeddings,spot_embeddings,loss1,gene_recon
    #def on_train_start(self):
    #    super().on_train_start()
    #    self.classification_head.to('cuda:2')

    #def _move_components_to_devices(self):
    #    super()._move_components_to_devices()
    #    self.classification_head.to('cuda:2')
    def training_step(self, batch, batch_idx):
        patch, target_gene, target_seg = batch
        pred_seg = self(patch)

        pred_seg=pred_seg.to('cuda:0')
        target_seg = target_seg.to(device=pred_seg.device, dtype=torch.long)  # ensure the target is long type
        pred_seg = pred_seg.view(-1, 3)  # [B*N, C]
        target_seg = target_seg.view(-1)  # [B*N]
        criterion = nn.CrossEntropyLoss()
        loss=criterion(pred_seg, target_seg)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_loss_seg', loss,on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        patch, target_gene, target_seg = batch
        pred_seg = self(patch)
        pred_seg=pred_seg.to('cuda:0')
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_seg, target_seg)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_loss_seg', loss, on_epoch=True)
        return loss
    def test_step(self, batch, batch_idx):
        patch, target_gene, target_seg = batch
        pred_seg = self(patch)
        pred_seg=pred_seg.to('cuda:0')
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_seg, target_seg)
        self.log('loss', loss,  prog_bar=True)
        self.log('test_loss_seg', loss)
        return loss
    #def configure_optimizers(self):
    #    optimizer = torch.optim.Adam(list(self.parameters()), lr=self.learning_rate)
    #    return optimizer


if __name__ == '__main__':
    a = torch.rand(1,4000,3*112*112)
    p = torch.ones(1,4000,2).long()
    model = HisToGene()
    print(count_parameters(model))
    x = model(a,p)
    print(x.shape)

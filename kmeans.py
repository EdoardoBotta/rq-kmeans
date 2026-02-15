import torch
from tqdm import tqdm
from kernels.quantize import kmeans_quantize
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Kmeans(nn.Module):
    def __init__(self, k, dim):
        super().__init__()

        self.k = k
        self.dim = dim
        
        self.register_buffer("centroids", torch.randn(k, dim))
        self.register_buffer("acc_cluster_emb", torch.zeros(k, dim))
        self.register_buffer("acc_cluster_size", torch.zeros(k))

    @torch.no_grad()
    def reset(self):
        self.centroids.normal_()
        self.acc_cluster_emb.zero_()
        self.acc_cluster_size.zero_()
    
    @torch.no_grad
    def update_centroids(self):
        update_mask = self.acc_cluster_size != 0

        acc_cluster_size = torch.where(update_mask, self.acc_cluster_size, 1.0)
        acc_cluster_emb = torch.where(update_mask.unsqueeze(1), self.acc_cluster_emb, self.centroids)
        
        update_norm = torch.nn.functional.pairwise_distance(self.centroids, acc_cluster_emb)
        self.centroids[:,:] = acc_cluster_emb / acc_cluster_size.unsqueeze(1)

        self.acc_cluster_emb.zero_()
        self.acc_cluster_size.zero_()

        return update_norm

    @torch.no_grad
    def fit(self, data: DataLoader, max_iters=10_000, tol=1e-6):
        for i in tqdm(range(max_iters)):
            for batch in data:
                batch = batch.to(self.centroids.device)
                self.forward(batch, acc=True)
            
            if (self.update_centroids().max() < tol).all():
                break
    
    @torch.no_grad
    def forward(self, x, acc=False):
        indices = kmeans_quantize(x, self.centroids)

        if acc:
            self.acc_cluster_emb.scatter_add_(0, indices.unsqueeze(1).expand(indices.shape[0], self.dim), x)
            self.acc_cluster_size.scatter_add_(0, indices, torch.ones_like(x[:, 0]))
        
        return self.centroids[indices], indices

class RQKmeans(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        n_layers: int = 3,
    ):
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            Kmeans(k=codebook_size, dim=dim) for _ in range(self.n_layers)
        ])
    
    @torch.no_grad
    def fit_codebooks(self, data: DataLoader, max_iters=10_000, tol=1e-6):
        for i, layer in enumerate(self.layers):
            for _ in range(max_iters):
                for batch in data:
                    batch = batch.to(self.centroids.device)
                    for pre_layer in self.layers[:i]:
                        emb, _ = pre_layer(batch, acc=False)
                        batch = batch - emb

                    layer(batch, acc=True)
                
                if (layer.update_centroids().max() < tol).all():
                    break
    
    def forward(self, x):
        ids = []
        for layer in self.layers:
            emb, batch_ids = layer(x)
            x = x - emb
            ids.append(batch_ids)
        return torch.stack(ids)


class RandomVectorDataset(Dataset):
    def __init__(self, size=10_000, dim=64):
        super().__init__()
        self.size = size
        self.data = torch.randn(size, dim)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
        

def train():
    size = 10_000
    k, dim = 5, 64
    batch_size = 32
    tol = 1e-4
    max_iters = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kmeans = Kmeans(k=k, dim=dim).to(device)
    dataset = RandomVectorDataset(size, dim)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    centroids = kmeans.fit(dataloader, max_iters=max_iters, tol=tol)
        

if __name__ == "__main__":
    train()
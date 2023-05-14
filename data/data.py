from torch.utils.data.dataset import Dataset

class ImageGenoDataset(Dataset):
    def __init__(self, images, modes, targets, transform):
        """
        images -- Batch size # of images
        snps -- SNP genotype matrix (0, 1, 2); (batch_size, num_genes)
        """
        self.images = images.type(torch.FloatTensor)
        self.modes = [mode.type(torch.FloatTensor) for mode in modes]
        self.targets = targets.type(torch.FloatTensor)
        self.transform = transform

    def __getitem__(self, index): 
        sample = self.images[index]
        modes = [mode[index] for mode in self.modes]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, modes, target
    
    def __len__(self):
        return len(self.images)
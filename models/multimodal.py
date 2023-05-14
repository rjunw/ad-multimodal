import torch.nn as nn

class MultimodalNet(nn.Module):
    def __init__(self, img_enc, mode_encs, mode_kwargs, latent_dim = 20, p = 0.3):
        """
        Multimodal network combining image encoder with other modalities

            img_enc -- Image encoder
            model_encs -- List of models encoding each modality
            mode_kwargs -- List of parameter dictionaries for each modality encoder
                mode_kwargs should not have an out_dim key, but every model should have 
                an out_dim argument
            latent_dim -- Dimension of shared latent space
        """
        self.im_model = img_enc(latent_dim, p)
        self.model = []
        for i in range(len(mode_encs)):
            self.model.append(mode_encs[i](**mode_kwargs[i], out_dim = latent_dim))

    def forward(self, im, modes):
        """
            im -- image tensor (batch_size x ...)
            modes -- list of mode batch tensors
        """
        z_im = self.im_model(im)
        z_modes = []
        for i in range(len(modes)):
            z_modes.append(self.model[i](modes[i]))
        
        return z_im, z_modes


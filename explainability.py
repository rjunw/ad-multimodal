import torch 
import torch.nn as nn
import tqdm
from captum.attr import IntegratedGradients
from utils.loss import NTXentLoss

class ImageExplainer(nn.Module):
    def __init__(self, model, ref_img, ref_modes, batch_size, w, device = 'cpu'): 
        """
        Forward pass of Explainer module returns self-supervised contrastive loss of 
        modalities on an image embedding.

        Closely following:
            https://github.com/HealthML/ContIG/blob/main/feature_explanations.py

        Params:
            model -- Multimodal neural network
            ref_img -- Reference image batch (contrastive loss works in batches)
            ref_modes -- Reference modality batch
                ref_modes[i] is a tensor representing the batch of observations for mode i
            batch_size -- Batch size for NTXent Loss
            w -- Weighting for loss
        """
        super().__init__() 
        self.device = device
        self.model = model
        self.model.eval()
        self.model.im_model.eval()
        for mode_model in model.model:
            mode_model.eval()

        with torch.no_grad(): 
            # set reference embeddings
            self.ref_img_embedding = self.model.im_model(ref_img.to(self.device))
            self.ref_modes_embedding = []

            # for each mode
            for i in range(len(ref_modes)):
                self.ref_modes_embedding.append(model.model[i](ref_modes[i]).to(self.device))

        self.loss_fn = NTXentLoss(device, 
                               batch_size, 
                               temperature = 0.07, 
                               alpha_weight = 0.75) # 0.75 weight towards images as in ContIG paper

        self.w = w

        def lf(imgE, embeddings):
            """
            Calculate loss for list of modalitie
            
                imgE -- image embedding
                embeddings -- List of (batch_size, mode features) tensors for each modality
            """
            loss = 0.0
            for modeE in embeddings:
                loss += self.w * self.loss_fn(imgE, modeE)
            return loss 
        
        self.loss = lf

    def set_img_embedding(self, img):
        """
        Image embedding that we want to explain
        """
        with torch.no_grad():
            self.img_embedding = (
                self.model.im_model(img.unsqueeze(0).to(self.device))
            ).to(self.device)

    def forward(self, modes):
        """
        Run the explainer. Returns list of losses for each individual.

            modes -- list of tensors 
                modes[i] is a tensor # individuals to explain x mode i feature dim
        """
        # run w/ modes associated w/ same image
        loss = torch.zeros(len(modes)) # one loss for each individual
        # print(modes.shape)

        # concatenate reference and current embeddings
        img_embedding_mod = torch.cat(
            [
                self.ref_img_embedding,
                self.img_embedding,
            ]
        ).to(self.device)
        # print(img_embedding_mod.shape)
        # print(modes.shape)

        # for each individual
        for i in range(len(modes)):
            # for each mode
            embeddings = []
            for j in range(modes.shape[-1]):
                embeddings.append(
                    torch.cat([
                        self.ref_modes_embedding[j],
                        self.model.model[j](modes[i, :, j].unsqueeze(0)).to(self.device)
                    ])
                )

            loss[i] = self.loss(
                img_embedding_mod,
                embeddings
            )

        return loss


def explain_image(im, modes, explainer, attr_kwargs = dict(), device = 'cpu', use_tqdm = True):
    """
    Explain an image using it's modes
    """
    attr_tests = [] # store attributes
    modes = torch.cat([mode.unsqueeze(-1) for mode in modes], axis = -1)
    print(modes.shape)

    # for each observation
    gen = tqdm(range(len(modes))) if use_tqdm else range(len(modes))
    for i in gen: 
        explainer.set_img_embedding(im[i].to(device))
        attr_test = IntegratedGradients(explainer).attribute(
            modes, **attr_kwargs
        )
        attr_tests.append(attr_test.detach().cpu().double().numpy())

    return attr_tests

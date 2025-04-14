from torch import nn


class CustomPredictor(nn.Module):
    def __init__(
            self,
            drug_encoder: nn.Module,
            protein_encoder: nn.Module,
            decoder: nn.Module,
    ):
        super().__init__()
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.decoder = decoder

    def forward(self, enc_drug, enc_protein):
        enc_drug = self.drug_encoder(enc_drug)
        enc_protein = self.protein_encoder(enc_protein)
        preds = self.decoder(enc_drug, enc_protein)

        return preds

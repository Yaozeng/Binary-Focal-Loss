class FocalLoss(nn.Module):
    def __init__(self, alpha=None,gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha=alpha
        
    def forward(self, input, target):
        per_entry_cross_ent = F.binary_cross_entropy_with_logits(input=input, target=target,reduce=False)
        prediction_probabilities = torch.sigmoid(input)
        p_t = (target * prediction_probabilities) +((1 - target) * (1 - prediction_probabilities))
        modulating_factor = 1.0
        if self.gamma:
            modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        alpha_weight_factor = 1.0
        if self.alpha is not None:
            alpha_weight_factor = target * self.alpha +(1 - target) * (1 - self.alpha)
        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor *per_entry_cross_ent
        return torch.sum(focal_cross_entropy_loss)

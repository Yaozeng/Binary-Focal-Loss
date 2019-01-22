#default alpha=None
#alpha:(1,C)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None,gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha=alpha
        
    def forward(self, input, target):
        per_entry_cross_ent = F.cross_entropy(input=input, target=target,reduce=False)
        prediction_probabilities = F.softmax(input)
        p_t = prediction_probabilities.gather(1,target).view(-1)
        modulating_factor = 1.0
        if self.gamma:
            modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        alpha_weight_factor = 1.0
        if self.alpha is not None:
            alpha_weight_factor = self.alpha.gather(0,target)
        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor *per_entry_cross_ent
        #return torch.sum(focal_cross_entropy_loss)
        return torch.mean(focal_cross_entropy_loss)

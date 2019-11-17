from torch.nn.functional import binary_cross_entropy_with_logits


class ObjectiveFunction:
    def __init__(self, target_adj):
        num_edges = target_adj.sum()
        num_nodes = target_adj.shape[0]
        self.pos_weight = float(num_nodes ** 2 - num_edges) / num_edges
        self.norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)
        self.target = target_adj

    def cal_loss(self, logit1, logit2, rho):
        loss1 = self.norm * binary_cross_entropy_with_logits(logit1, self.target, pos_weight=self.pos_weight,
                                                             reduction='mean')

        loss2 = self.norm * binary_cross_entropy_with_logits(logit2, self.target, pos_weight=self.pos_weight,
                                                             reduction='mean')

        return loss1 + rho * loss2


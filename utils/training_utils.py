
import numpy as np
import torch as th

from transformers import Trainer


def compute_class_weights(mlb_labels):
    occ_ratios = (mlb_labels.sum() / mlb_labels.sum(axis=0))
    occ_ratios /= occ_ratios.min()
    occ_ratios = np.power(occ_ratios, 1 / 3)

    class_weights = dict(zip(np.arange(mlb_labels.shape[1]), occ_ratios))

    return class_weights


class ImbalancedTrainer(Trainer):
    def __init__(self, class_weights,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You pass the class weights when instantiating the Trainer
        self.class_weights = th.Tensor(list(class_weights.values())).cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.

            # Changes start here
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            logits = outputs['logits']
            criterion = FocalLoss(self.class_weights)
            loss = criterion(logits, inputs['labels'])
            # Changes end here

        return (loss, outputs) if return_outputs else loss


class FocalLoss(th.nn.Module):
    def __init__(self, pos_weight, alpha=0.1, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight.to('cuda')

    def forward(self, inputs, targets):
        BCE_loss = th.nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)(inputs, targets)
        pt = th.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

from __future__ import annotations
import warnings
import torch
from monai.networks.layers.utils import get_act_layer
from monai.utils import LossReduction
from monai.utils.enums import StrEnum
from torch.nn.modules.loss import _Loss


class AdversarialCriterions(StrEnum):
    BCE = "bce"
    HINGE = "hinge"
    LEAST_SQUARE = "least_squares"


class PatchAdversarialLoss(_Loss):
    """"""
    def __init__(self, reduction: LossReduction | str = LossReduction.MEAN,
                 criterion: str = AdversarialCriterions.LEAST_SQUARE.value, # criterion="least_squares"
                 no_activation_leastsq: bool = False,) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if criterion.lower() not in [m.value for m in AdversarialCriterions]:
            raise ValueError("Unrecognised criterion entered for Adversarial Loss. Must be one in: %s"
                             % ", ".join([m.value for m in AdversarialCriterions]))
        # Depending on the criterion, a different activation layer is used.
        self.real_label = 1.0
        self.fake_label = 0.0

        if criterion == AdversarialCriterions.BCE.value:
            self.activation = get_act_layer("SIGMOID")
            self.loss_fct = torch.nn.BCELoss(reduction=reduction)

        elif criterion == AdversarialCriterions.HINGE.value:
            self.activation = get_act_layer("TANH")
            self.fake_label = -1.0

        elif criterion == AdversarialCriterions.LEAST_SQUARE.value:
            if no_activation_leastsq:
                self.activation = None
            else:
                self.activation = get_act_layer(name=("LEAKYRELU", {"negative_slope": 0.05}))
            self.loss_fct = torch.nn.MSELoss(reduction=reduction)

        self.criterion = criterion
        self.reduction = reduction

    def get_target_tensor(self, input: torch.FloatTensor, target_is_real: bool) -> torch.Tensor:

        filling_label = self.real_label if target_is_real else self.fake_label
        label_tensor = torch.tensor(1).fill_(filling_label).type(input.type()).to(input[0].device)
        label_tensor.requires_grad_(False)
        return label_tensor.expand_as(input)

    def get_zero_tensor(self, input: torch.FloatTensor) -> torch.Tensor:
        """
        Gets a zero tensor.

        Args:
            input: tensor which shape you want the zeros tensor to correspond to.
        Returns:
        """

        zero_label_tensor = torch.tensor(0).type(input[0].type()).to(input[0].device)
        zero_label_tensor.requires_grad_(False)
        return zero_label_tensor.expand_as(input)

    def forward(self,
                input: torch.FloatTensor | list,
                target_is_real: bool,
                for_discriminator: bool) -> torch.Tensor | list[torch.Tensor]:
        if not for_discriminator and not target_is_real:
            target_is_real = True  # With generator, we always want this to be true!
            warnings.warn("Variable target_is_real has been set to False, but for_discriminator is set"
                          "to False. To optimise a generator, target_is_real must be set to True.")
        if type(input) is not list:
            input = [input]
        target_ = []
        for i, disc_out in enumerate(input):
            if self.criterion != AdversarialCriterions.HINGE.value:
                trg_attention_tensor = self.get_target_tensor(disc_out, target_is_real)
                target_.append(trg_attention_tensor)
                if target_is_real == False :
                    print(f'target_is_real == False (zero) : {trg_attention_tensor}')
            else:
                target_.append(self.get_zero_tensor(disc_out))

        # Loss calculation
        loss = []
        for disc_ind, disc_out in enumerate(input):

            if self.activation is not None:
                disc_out = self.activation(disc_out)

            if self.criterion == AdversarialCriterions.HINGE.value and not target_is_real:
                loss_ = self.forward_single(-disc_out, target_[disc_ind])

            else:
                loss_ = self.forward_single(disc_out, target_[disc_ind])
            loss.append(loss_)

        if loss is not None:
            if self.reduction == LossReduction.MEAN.value:
                loss = torch.mean(torch.stack(loss))
            elif self.reduction == LossReduction.SUM.value:
                loss = torch.sum(torch.stack(loss))

        return loss

    def forward_single(self,
                       input: torch.FloatTensor,
                       target: torch.FloatTensor) -> torch.Tensor | None:

        if (self.criterion == AdversarialCriterions.BCE.value or self.criterion == AdversarialCriterions.LEAST_SQUARE.value):
            # self.loss_fct .. ?
            # self.loss_fct = torch.nn.MSELoss(reduction=reduction)
            # mse loss
            return self.loss_fct(input, target)

        elif self.criterion == AdversarialCriterions.HINGE.value:
            minval = torch.min(input - 1, self.get_zero_tensor(input))
            return -torch.mean(minval)

        else:
            return None
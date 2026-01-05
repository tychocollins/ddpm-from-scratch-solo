import torch
import copy

class EMA:
    """
    Exponential Moving Average handler for model weights.
    Provides the stabilization required for high-fidelity generation.
    """
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        # Create a deep copy of the model to store the averaged weights
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        # Ensure no gradients are calculated for the EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA weights using the current model weights.
        Formula: ema_weight = decay * ema_weight + (1 - decay) * current_weight
        """
        with torch.no_grad():
            current_params = dict(model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())

            for name, param in current_params.items():
                # Linearly interpolate between current and EMA weights
                ema_params[name].copy_(
                    self.decay * ema_params[name] + (1.0 - self.decay) * param.data
                )

    def get_ema_model(self):
        """Returns the model with stabilized weights."""
        return self.ema_model
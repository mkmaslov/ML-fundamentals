import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader, TensorDataset


class TorchMLPNative(nn.Module):

    def __init__(
        self,
        dim_input,
        dim_output,
        activation_function,
        activation_function_out,
        optimizer,
        dropout_rate,
        num_layers: int = 2,
        dim_hidden: int = 16,
        kaiming_bias: float  = 0.01,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> None: 
        
        """Implementation of a multi-layer perceptron (MLP) using PyTorch."""
        
        # Inherit settings from parent class.
        super().__init__()
        # Create array of dimensions.
        dims = [dim_input] + [dim_hidden]*(num_layers-1) + [dim_output]
        # Create empty list for layers.
        layers: list[nn.Module] = []

        # Go through layers and initialize structure within each layer.
        for layer in range(num_layers):
            # Linear layer = affine transform (A*x+B)
            layers.append(nn.Linear(dims[layer], dims[layer+1], \
                bias=True, device=device, dtype=dtype))
            if layer != num_layers-1:
                if activation_function=="relu":
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_rate))
        self.net = nn.Sequential(*layers)
        self.reset_parameters(bias = kaiming_bias)
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype

    def reset_parameters(self, bias: float = 0.01):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, \
                    mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _make_optimizer(self,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001):
        if self.optimizer == "SGD":
            return torch.optim.SGD(self.parameters(), \
                lr = lr, momentum = momentum, weight_decay = weight_decay)
        elif self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), \
                lr = lr, weight_decay = weight_decay)

    @torch.no_grad()
    def _accuracy(self, logits: torch.Tensor, y_true: torch.Tensor, nn_type: str):
        if nn_type == "classifier":
            y_pred = logits.argmax(dim=1)
            return (y_pred == y_true).float().mean().item()

    # TODO: what happens to parameters?? Re-init every time?
    def train_supervised(
        self, 
        inputs: torch.Tensor,
        targets: torch.Tensor, 
        nn_type: str = "classifier",
        lr: float = 0.05,
        num_epochs: int = 300, 
        batch_size: int = 128,
        log_step: int = 10,
        weight_decay: float = 0.0001,
        momentum: float = 0.9,
    ):
        """
        Training routine for supervised learning.

        Input
        -----
        inputs: torch.Tensor
            Shape: (NUM_SAMPLES, NUM_FEATURES)
        data_train: torch.Tensor
            Shape: (NUM_SAMPLES, )
        parameters: tuple[tuple[torch.Tensor, torch.Tensor]]
            Weights and biases for all layers.
        rate_init: float
            Initialization value of the learning rate.
        num_epochs: int
            Number of epochs for training.
        dim_batch: int
            Batch size for training.
        lambda_reg: float
            L2 regularization parameter.
        momentum: float
            Parameter for momentum gradient method.
        """

        # Move tensors to target device
        x = inputs.to(device=self.device, dtype=self.dtype)
        if nn_type == "classifier":
            y = targets.to(device=self.device, dtype=torch.long)
            loss_function = nn.CrossEntropyLoss()

        # Create an iterator over batches
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

        optimizer = self._make_optimizer(lr=lr, \
            momentum=momentum, weight_decay=weight_decay)

        epochs: list[int] = []
        losses: list[float] = []
        accuracies: list[float] = []

        # Iterate over epochs
        for epoch in range(num_epochs + 1):
            
            # Switches model into the training mode
            self.train()

            # Iterate over batches
            for xb, yb in loader:

                # Reset gradients
                optimizer.zero_grad(set_to_none=True)

                # Apply forward propagation and calculate loss
                out = self.forward(xb)
                loss = loss_function(out, yb)

                # Evaluate backward propagation using autograd engine
                loss.backward()

                # Apply one optimizer step - gradient decent
                optimizer.step()

            # Print out log
            if (epoch % log_step == 0) or (epoch == num_epochs):
                # Switch off training mode
                self.eval()
                # Turn off autograd to save resources
                with torch.no_grad():
                    out_total = self.forward(x)
                    loss_total = loss_function(out_total, y).item()
                    accuracy_total = self._accuracy(out_total, y, nn_type)
                # Print out status message
                msg = f"\rEpoch {epoch}/{num_epochs}: loss {loss_total:.4f}, accuracy {accuracy_total:.4f}"
                if epoch==num_epochs: msg+="\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                # Save data
                epochs.append(epoch)
                losses.append(loss_total)
                accuracies.append(accuracy_total)

        return epochs, losses, accuracies

    @torch.no_grad()
    def validate(self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        nn_type: str = "classifier"):
        """
        Validate network's performance on a test dataset.
        """
        # Switch off training mode
        self.eval()
        # Move data to torch.device
        x = inputs.to(self.device, dtype=self.dtype)
        y = targets.to(self.device, dtype=torch.long)
        # Perform forward propagation
        logits = self.forward(x)
        # Calculate loss
        loss = nn.CrossEntropyLoss()(logits, y).item()
        # Calcualte accuracy
        accuracy = self._accuracy(logits, y, nn_type)
        # Print out status message.
        print(f"Validation loss {loss:.4f}, accuracy {accuracy:.4f}")
        return loss, accuracy

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor):
        """Predict classes for inputs."""
        # Switch off training mode
        self.eval()
        # Move data to torch.device
        x = inputs.to(device=self.device, dtype=self.dtype)
        # Perform forward propagation
        logits = self.forward(x)
        return logits.argmax(dim=1)
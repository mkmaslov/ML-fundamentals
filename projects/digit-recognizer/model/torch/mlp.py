import torch, sys
from typing import Callable


class TorchMLP:

    def __init__(
        self,
        dim_input: int = None,
        dim_output: int = None,
        num_layers: int = 2,
        dim_hidden: int = 16,
        activation_function: str = "ReLU",
        activation_function_out: str = "softmax",
        optimizer: str = "adam",
        dropout_rate: float = 0.1,
        model_type: str = "classifier",
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> None: 
        """Implementation of a multi-layer perceptron (MLP) using PyTorch."""
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.model_type = model_type
        self.dtype = dtype
        self.device = device
        self.dropout_rate = None
        self.dropout_on = False
        self.parameters = None

    def initialize_layer(
        self,
        dim_in: int, 
        dim_out: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Kaiming-initialized weight and bias for a single layer."""
        gain = (2.0 / dim_in)**0.5
        W = gain * \
            torch.randn(dim_in, dim_out, dtype=self.dtype, device=self.device)
        b = torch.full((1,dim_out), 0.01, dtype=self.dtype, device=self.device)
        return W, b

    def initialize_network(
        self,
        dim_in: int,
        dim_out: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        """Initialize weights and biases for all layers."""
        dims = [dim_in] + [self.dim_hidden]*(self.num_layers-1) + [dim_out] 
        return tuple(self.initialize_layer(dim_in, dim_out)
                for dim_in, dim_out in zip(dims[:-1], dims[1:]))

    # ACTIVATION FUNCTIONS:
    # Transform logits (Z) into outputs (Y).

    def relu(self, Z: torch.Tensor) -> torch.Tensor:
        """ReLU (rectified linear unit): removes negative values."""
        return torch.maximum(Z,Z.new_zeros(()))
    def relu_grad(self, Z: torch.Tensor) -> torch.Tensor:
        """Derivative of the ReLU function."""
        return (Z>0).type_as(Z)

    def tanh(self, Z: torch.Tensor) -> torch.Tensor:
        """tanh (Hyperbolic tangent): continuous activation function."""
        return torch.tanh(Z)
    def tanh_grad(self, Z: torch.Tensor) -> torch.Tensor:
        """Derivative of the tanh function."""
        return 1.0 - torch.tanh(Z)**2
       
    def softmax(self, Z: torch.Tensor) -> torch.Tensor:
        """Final layer for classifiers: convert logits into probabilities."""
        exp_z = torch.exp(Z - Z.max(dim=1, keepdim=True).values)
        return exp_z / exp_z.sum(dim=1, keepdim=True)

    def dropout(self, X: torch.Tensor) -> torch.Tensor:
        """Inverted dropout: drop a unit with probability (dropout_rate)."""
        if (not self.dropout_on) or self.dropout_rate == 0.0:
            return X
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError("Dropout probability must be in [0, 1).")
        keep = 1.0 - self.dropout_rate
        mask = torch.rand_like(X) < keep
        return X * mask.to(X.dtype) / keep

    # LOSS FUNCTIONS:

    def cross_entropy(
        self,
        Z_pred: torch.Tensor,
        Y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Logarithm of the probability to predict correct classes for all samples.
        """
        Z_stable = Z_pred - Z_pred.max(dim=1, keepdim=True).values
        logsumexp = Z_stable.exp().sum(dim=1, keepdim=True).log()
        # log_probs is numerically-stable equivalent to log(softmax(Z_pred))
        log_probs = Z_stable - logsumexp
        true_logp = log_probs.gather(1, Y_true.unsqueeze(1)).squeeze(1)
        return -true_logp.mean()

    def cross_entropy_grad(
        self,
        probs_pred: torch.Tensor,
        Y_true: torch.Tensor
    ) -> torch.Tensor:
        """Derivative of the cross_entropy function.
        
        Uses probs_pred instead of Z_pred to save time.
        """
        N = probs_pred.size(0)
        g = probs_pred.clone()
        g[torch.arange(N, device=probs_pred.device), Y_true] -= 1
        return g / N

    # FORWARD PROPAGATION

    def forward_layer(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate forward propagation at a single level."""
        # Calculate arbitrary-valued scores (logits).
        Z = X @ W + b
        # Apply the activation function - calculate the outputs.
        Y = activation_function(Z)
        # Apply the dropout.
        Y = self.dropout(Y)
        return Z, Y

    def forward(
        self,
        inputs: torch.Tensor,
        parameters: tuple[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate forward propagation at all levels.
        Input
        -----
        X_init: torch.Tensor
        parameters : tuple[tuple[torch.Tensor, torch.Tensor]]
            Weights and biases at all layers.
        Output
        ------
        forward_ML : tuple[tuple[torch.Tensor, torch.Tensor]]
            For every layer, scores before and after applying the activation function.
        """
        forward_parameters, X = [[0, inputs]], inputs
        # For every layer: calculate forward propagation and update initial vector
        for index, parameter in enumerate(parameters):
            # softmax is only applied at the outer level
            activation_function = self.relu \
                if index!=(self.num_layers-1) else self.softmax
            forward_parameter = self.forward_layer(X,*parameter,activation_function)
            X = forward_parameter[1]
            forward_parameters.append(forward_parameter)
        # Export list of [Z, Y] for all layers:
        # index 0 corresponds to the initial data, so len(forward_params)=num_layers+1
        return forward_parameters

    # BACKWARD PROPAGATION

    # Single-layer backward propagation
    # Examples:
    # params = [cross_entropy_derivative, probs_true]
    # params = [relu_derivative, W_next, dL_dZ_next]
    def backward_layer(self, X, Z, backward_parameters):
        layer = backward_parameters[0]
        # Cross-entropy is always the outer layer: probs_pred = Z
        if layer==self.num_layers-1:
            #probs_pred, probs_true = backward_parameters[1], backward_parameters[2]
            #dL_dZ = self.cross_entropy_grad(probs_pred, probs_true)
            probs_pred, Y_true = backward_parameters[1], backward_parameters[2]
            dL_dZ = self.cross_entropy_grad(probs_pred, Y_true)
        # ReLU is always an inner layer, so it needs:
        # W_next and dL_dZ_next from the "next" (in terms of forward propagation) step
        else:
            W_next, dL_dZ_next = backward_parameters[1:]
            # Current step's Y is "next" step's X
            dL_dY = dL_dZ_next @ W_next.T
            dL_dZ = dL_dY * self.relu_grad(Z)
        # This works for every layer: Z = W*X + b
        dL_dW = X.T @ dL_dZ
        dL_db = dL_dZ.sum(0, keepdim=True)
        # Export derivatives
        return [dL_dZ, dL_dW, dL_db]

    # Multi-layer backward propagation
    def backward(
        self,
        X_init,
        Y_true,
        parameters,
        forward_parameters
    ):
        grads = [[None, None, None] for _ in range(self.num_layers)]
        # Layers are counted in the direction of forward propagation
        # layer 0 is the initial data
        for layer in range(self.num_layers-1,-1,-1):
            # If it is the outer layer, apply cross_entropy_derivative
            # If it is an inner layer, apply relu_derivative with:
            # W_next = parameters[layer-1,1]
            # dL_dZ_next = derivatives[0,0]
            backward_parameters = \
                [layer, forward_parameters[-1][1], Y_true] \
                if layer==self.num_layers-1 else \
                [layer, parameters[layer+1][0], grads[layer+1][0]]
            X, Z = forward_parameters[layer][1], forward_parameters[layer+1][0]
            grads[layer] = self.backward_layer(X, Z, backward_parameters)
        return grads

    def train_supervised(
        self, 
        inputs: torch.Tensor,
        targets: torch.Tensor, 
        nn_type: str = "classifier",
        lr: float = 0.05,
        num_epochs: int = 300, 
        batch_size: int = 128,
        log_step: int = 10,
        lambda_reg: float = 0.0001,
        momentum: float = 0.9,
        dropout_rate: float = 0.1,
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
        lr: float
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

        # Determine dimensions
        num_samples = inputs.shape[0]
        self.dim_input = inputs.shape[1]
        if self.model_type=="classifier":
            self.dim_output = int(targets.max().item()) + 1
        else:
            raise ValueError(f"Unknown model_type={self.model_type!r}")

        # Turn on the dropout
        self.dropout_rate = dropout_rate
        self.dropout_on = True

        # Assign default parameters
        if self.parameters is None:
            parameters_init = self.initialize_network(self.dim_input, self.dim_output)
            self.parameters = [[W.clone(), b.clone()] for (W, b) in parameters_init]

        # Initialize velocities
        # velocities = [[v_W1,v_b1],[v_W2,v_b2],...]
        velocities = [[torch.zeros_like(W), torch.zeros_like(b)] for (W, b) in self.parameters]

        # Iterate over epochs
        epochs = []
        losses = []
        accuracies = []
        for epoch in range(num_epochs + 1):

            # Calculate gradient descents within small batches of entire dataset.
            for batch_start in range(0, num_samples, batch_size):
                
                batch_end = batch_start + batch_size
                batch_inputs = inputs[batch_start:batch_end]
                batch_Y_true = targets[batch_start:batch_end]

                # Forward propagation
                forward_parameters = self.forward(batch_inputs, self.parameters)

                # Backward propagation
                grads = self.backward(batch_inputs, batch_Y_true, \
                    self.parameters, forward_parameters)

                # Iterate over layers
                for layer in range(self.num_layers):
                    # L2-regularization
                    grads[layer][1] += lambda_reg*self.parameters[layer][0]
                    # Re-calculate velocities
                    velocities[layer][0] = \
                        momentum*velocities[layer][0] - lr*grads[layer][1]
                    velocities[layer][1] = \
                        momentum*velocities[layer][1] - lr*grads[layer][2]
                    # Perform gradient descent
                    self.parameters[layer][0] += velocities[layer][0]
                    self.parameters[layer][1] += velocities[layer][1]

            # Print out loss function values.
            if (epoch % log_step == 0) or (epoch==num_epochs):

                # Calculate forward propagation on the entire dataset.
                self.dropout_on = False
                data_forward = self.forward(inputs, self.parameters)
                self.dropout_on = True
                Z_pred, probs_pred = data_forward[-1][0], data_forward[-1][1]
                # Calculate loss.
                loss = self.cross_entropy(Z_pred, targets).item()

                # Calculate accuracy.
                accuracy = (probs_pred.argmax(1) == targets).float().mean().item()
                # Print out status message.
                msg = f"\rEpoch {epoch}/{num_epochs}: loss {loss:.4f}, accuracy {accuracy:.4f}"
                if epoch==num_epochs: msg+="\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                # Save data.
                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(accuracy)

        return epochs, losses, accuracies

    def validate(
        self,
        inputs: torch.Tensor,
        data_test: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Validate network's performance on a test dataset.
        """
        # Turn off the dropout
        self.dropout_on = False
        # Perform forward propagation.
        forward_parameters = self.forward(inputs, self.parameters)
        logits_pred, probs_pred = forward_parameters[-1][0], forward_parameters[-1][1]
        # Calculate loss.
        loss = self.cross_entropy(logits_pred, data_test).item()
        # Calculate accuracy.
        accuracy = (probs_pred.argmax(1) == data_test).float().mean().item()
        # Print out status message.
        print(f"Validation loss {loss:.4f}, accuracy {accuracy:.4f}")
        return loss, accuracy

    def predict(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict classes for inputs."""
        # Turn off the dropout
        self.dropout_on = False
        # Perform forward propagation.
        forward_parameters = self.forward(inputs, self.parameters)
        probs_pred = forward_parameters[-1][1]
        return probs_pred.argmax(1)

import warnings

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn

from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from e3nn.util.jit import compile_mode
from torch_scatter import scatter


def get_o3_swish_gate_irreps(irreps: Irreps):

    # See https://github.com/RobDHess/Steerable-E3-GNN/blob/main/models/segnn/o3_building_blocks.py#L152
    irreps_g_scalars = Irreps(str(irreps[0]))
    irreps_g_gate = Irreps(
        "{}x0e".format(irreps.num_irreps - irreps_g_scalars.num_irreps)
    )
    irreps_g_gated = Irreps(str(irreps[1:]))

    irreps_dict = {
        "merged": (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify(),
        "separate": (irreps_g_scalars, irreps_g_gate, irreps_g_gated),
    }

    return irreps_dict


"""
The following classes (O3TensorProduct and O3SwishGate) are copied and adapted from "E(3) Steerable GNN" on GitHub
(https://github.com/RobDHess/Steerable-E3-GNN).
"""

class O3TensorProduct(nn.Module):
    """A bilinear layer, computing CG tensorproduct and normalising them.

    Parameters
    ----------
    irreps_in1 : o3.Irreps
        Input irreps.
    irreps_out : o3.Irreps
        Output irreps.
    irreps_in2 : o3.Irreps
        Second input irreps.
    tp_rescale : bool
        If true, rescales the tensor product.

    """

    def __init__(
        self, irreps_in1, irreps_out, irreps_in2=None, tp_rescale=True
    ) -> None:
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2
        if irreps_in2 == None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2
        self.tp_rescale = tp_rescale

        # Build the layers
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            shared_weights=True,
            irrep_normalization="component",  # [CHANGED] deprecated keyword argument
        )

        # For each zeroth order output irrep we need a bias
        # So first determine the order for each output tensor and their dims
        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_slices = irreps_out.slices()
        # Store tuples of slices and corresponding biases in a list
        self.biases = []
        self.biases_slices = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = irreps_out.slices()[slice_idx]
                out_bias = torch.zeros(
                    self.irreps_out_dims[slice_idx], dtype=self.tp.weight.dtype
                )
                self.biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]

        # Initialize the correction factors
        self.slices_sqrt_k = {}

        # Initialize similar to the torch.nn.Linear
        self.tensor_product_init()
        # Adapt parameters so they can be applied using vector operations.
        self.vectorise()

    def tensor_product_init(self) -> None:
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                slice_idx = instr[2]
                mul_1, mul_2, mul_out = weight.shape
                fan_in = mul_1 * mul_2
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            # Do the initialization of the weights in each instruction
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                slice_idx = instr[2]
                if self.tp_rescale:
                    sqrt_k = 1 / sqrt(slices_fan_in[slice_idx])
                else:
                    sqrt_k = 1.0
                weight.data.uniform_(-sqrt_k, sqrt_k)
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            # Initialize the biases
            for out_slice_idx, out_slice, out_bias in zip(
                self.biases_slice_idx, self.biases_slices, self.biases
            ):
                sqrt_k = 1 / sqrt(slices_fan_in[out_slice_idx])
                out_bias.uniform_(-sqrt_k, sqrt_k)

    def vectorise(self):
        """Adapts the bias parameter and the sqrt_k corrections so they can be applied using vectorised operations"""

        # Vectorise the bias parameters
        if len(self.biases) > 0:
            with torch.no_grad():
                self.biases = torch.cat(self.biases, dim=0)
            self.biases = nn.Parameter(self.biases)

            # Compute broadcast indices.
            bias_idx = torch.LongTensor()
            for slice_idx in range(len(self.irreps_out_orders)):
                if self.irreps_out_orders[slice_idx] == 0:
                    out_slice = self.irreps_out.slices()[slice_idx]
                    bias_idx = torch.cat(
                        (
                            bias_idx,
                            torch.arange(out_slice.start, out_slice.stop).long(),
                        ),
                        dim=0,
                    )

            self.register_buffer("bias_idx", bias_idx, persistent=False)
        else:
            self.biases = None

        # Now onto the sqrt_k correction
        sqrt_k_correction = torch.zeros(self.irreps_out.dim)
        for instr in self.tp.instructions:
            slice_idx = instr[2]
            slice, sqrt_k = self.slices_sqrt_k[slice_idx]
            sqrt_k_correction[slice] = sqrt_k

        # Make sure bias_idx and sqrt_k_correction are on same device as module
        self.register_buffer("sqrt_k_correction", sqrt_k_correction, persistent=False)

    def forward_tp_rescale_bias(self, data_in1, data_in2=None) -> torch.Tensor:
        if data_in2 == None:
            data_in2 = torch.ones_like(data_in1[:, 0:1])

        data_out = self.tp(data_in1, data_in2)

        # Apply corrections
        if self.tp_rescale:
            data_out /= self.sqrt_k_correction

        # Add the biases
        if self.biases is not None:
            data_out[:, self.bias_idx] += self.biases
        return data_out

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        return data_out


class O3SwishGate(torch.nn.Module):
    def __init__(self, irreps_g_scalars, irreps_g_gate, irreps_g_gated) -> None:
        super().__init__()
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(
                irreps_g_scalars,
                [nn.SiLU()],
                irreps_g_gate,
                [torch.sigmoid],
                irreps_g_gated,
            )
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in) -> torch.Tensor:
        data_out = self.gate(data_in)
        return data_out


"""
The following class (BatchNorm) is copied and adapted from "Euclidean neural networks" on GitHub (https://github.com/e3nn/e3nn).
"""

@compile_mode("unsupported")
class BatchNorm(nn.Module):
    """Batch normalization for orthonormal representations

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Parameters
    ----------
    irreps : `o3.Irreps`
        representation

    eps : float
        avoid division by zero when we normalize by the variance

    momentum : float
        momentum of the running average

    affine : bool
        do we have weight and bias parameters

    reduce : {'mean', 'max'}
        method used to reduce

    instance : bool
        apply instance norm instead of batch norm
    """

    def __init__(
        self,
        irreps,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        reduce="mean",
        instance=False,
        normalization="component",
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps

        if self.instance:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
        else:
            self.register_buffer("running_mean", torch.zeros(num_scalar))
            self.register_buffer("running_var", torch.ones(num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update.detach()

    def forward(self, input, batch_id_tensor=None):  # [CHANGED] include batch ID tensor
        """evaluate

        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        batch, *size, dim = input.shape
        input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # [ADDED] dummy batch ID tensor
        if batch_id_tensor is None:
            batch_id_tensor = torch.zeros(
                input.size(0), dtype=torch.int64, device=input.device
            )
        # [ADDED] ^^^

        if self.training and not self.instance:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:
            d = ir.dim
            field = input[:, :, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            # [batch, sample, mul, repr]
            field = field.reshape(batch, -1, mul, d)

            if ir.is_scalar():  # scalars
                if self.training or self.instance:
                    if self.instance:
                        # [CHANGED] batched instance mean
                        field_mean = scatter(
                            field.mean(1).reshape(batch, mul),
                            batch_id_tensor,
                            dim=0,
                            reduce="mean",
                        )[
                            batch_id_tensor
                        ]  # [batch, mul]
                        # [CHANGED] ^^^
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(
                            self._roll_avg(
                                self.running_mean[irm : irm + mul], field_mean
                            )
                        )
                else:
                    field_mean = self.running_mean[irm : irm + mul]
                irm += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if self.training or self.instance:
                if self.normalization == "norm":
                    field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
                elif self.normalization == "component":
                    field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError(
                        "Invalid normalization option {}".format(self.normalization)
                    )

                if self.reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif self.reduce == "max":
                    field_norm = field_norm.max(1).values  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                if not self.instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(
                        self._roll_avg(self.running_var[irv : irv + mul], field_norm)
                    )
                # [ADDED] batched instance norm
                elif self.instance:
                    field_norm = scatter(
                        field_norm, batch_id_tensor, dim=0, reduce="mean"
                    )[
                        batch_id_tensor
                    ]  # [batch, mul]
                # [ADDED] ^^^
            else:
                field_norm = self.running_var[irv : irv + mul]
            irv += mul

            field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[iw : iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [(batch,) mul]

            field = field * field_norm.reshape(
                -1, 1, mul, 1
            )  # [batch, sample, mul, repr]

            if self.affine and ir.is_scalar():  # scalars
                bias = self.bias[ib : ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch, sample, mul, repr]

            fields.append(
                field.reshape(batch, -1, mul * d)
            )  # [batch, sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        if self.training and not self.instance:
            assert irm == self.running_mean.numel()
            assert irv == self.running_var.size(0)
        if self.affine:
            assert iw == self.weight.size(0)
            assert ib == self.bias.numel()

        if self.training and not self.instance:
            if len(new_means) > 0:
                torch.cat(new_means, out=self.running_mean)
            if len(new_vars) > 0:
                torch.cat(new_vars, out=self.running_var)

        output = torch.cat(fields, dim=2)  # [batch, sample, stacked features]
        return output.reshape(batch, *size, dim)


class SEMLP(torch.nn.Module):
    def __init__(
        self,
        irreps: tuple,
        edge_attr_irreps=None,
        plain_last: bool = True,
        use_norm_in_first: bool = True,
    ):
        super().__init__()

        self.linear_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        # Catch annoying warnings
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)

            self.linear_layers.append(
                O3TensorProduct(
                    irreps[0],
                    get_o3_swish_gate_irreps(irreps[1])["merged"],
                    edge_attr_irreps,
                )
            )
            self.norm_layers.append(
                BatchNorm(get_o3_swish_gate_irreps(irreps[1])["merged"])
                if use_norm_in_first
                else torch.nn.Identity()
            )
            self.activations.append(
                O3SwishGate(*get_o3_swish_gate_irreps(irreps[1])["separate"])
            )

            for irreps_in, irreps_out in zip(irreps[1:-2], irreps[2:-1]):
                self.linear_layers.append(
                    O3TensorProduct(
                        irreps_in,
                        get_o3_swish_gate_irreps(irreps_out)["merged"],
                        edge_attr_irreps,
                    )
                )
                self.norm_layers.append(
                    BatchNorm(get_o3_swish_gate_irreps(irreps_out)["merged"])
                )
                self.activations.append(
                    O3SwishGate(*get_o3_swish_gate_irreps(irreps_out)["separate"])
                )

            if plain_last:
                self.linear_layers.append(
                    O3TensorProduct(*irreps[-2:], edge_attr_irreps)
                )
                self.norm_layers.append(torch.nn.Identity())
                self.activations.append(torch.nn.Identity())
            else:
                self.linear_layers.append(
                    O3TensorProduct(
                        irreps[-2],
                        get_o3_swish_gate_irreps(irreps[-1])["merged"],
                        edge_attr_irreps,
                    )
                )
                self.norm_layers.append(
                    BatchNorm(get_o3_swish_gate_irreps(irreps[-1])["merged"])
                )
                self.activations.append(
                    O3SwishGate(*get_o3_swish_gate_irreps(irreps[-1])["separate"])
                )

    def forward(
        self, x: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        for linear_layer, norm_layer, activation in zip(
            self.linear_layers, self.norm_layers, self.activations
        ):
            x = activation(norm_layer(linear_layer(x, edge_attr)))

        return x

"""TVnet regression network.

Mirrors ``legacy/TV tracking/network_code/ResNet50_chs.m`` (SPEC.md section 4).

The MATLAB script builds a ``layerGraph`` from MATLAB's ImageNet ResNet-50 as follows::

    lgraph = addLayers(lgraph, imageInputLayer([row,col,chann],'Name','input_1'));
    lgraph = addLayers(lgraph, convolution2dLayer(2,64,'Stride',1,'padding','same','Name','conv1'));
    for i=3:174                      % bn_conv1 ... avg_pool of MATLAB's resnet50
        lgraph = addLayers(lgraph, resnet.Layers(i));
    end
    lgraph = addLayers(lgraph, fullyConnectedLayer(1000,'Name','fc1000'));
    lgraph = addLayers(lgraph, fullyConnectedLayer(pts,'Name','fc6'));
    lgraph = addLayers(lgraph, regressionLayer('Name','output'));

Key consequences, reproduced here:

* MATLAB layer 2 (``conv1``, the pretrained 7x7 / stride-2 stem) is **not** copied.  It is
  replaced by a brand-new 2x2 / stride-1 / 'same' convolution, so the first activation keeps
  the *full* input resolution and every downstream feature map is 2x the usual ResNet-50
  spatial size (160x160 in, 160x160 after the stem, 80x80 after max-pool).
* Layers 3..174 are MATLAB's pretrained ``bn_conv1`` ... ``avg_pool`` — i.e. batch-norm,
  ReLU, max-pool, the four residual stages and global average pooling.  Their ImageNet
  weights are kept.  Here they come from ``torchvision`` ``ResNet50_Weights.IMAGENET1K_V1``
  (MATLAB's ``resnet50`` is the same Keras-derived ImageNet model; the numeric weights are
  not bit-identical between frameworks, which is unavoidable and does not change the graph).
* MATLAB's ``fullyConnectedLayer`` is *affine only* — no activation, no dropout.  The
  default ``head='matlab'`` is therefore ``Linear(2048, 1000) -> Linear(1000, n_out)`` with
  nothing in between.
* ``regressionLayer`` is plain MSE on the raw pixel coordinates.  The loss is the caller's
  job (see SPEC.md section 4: "Targets are raw pixel coordinates in the network's input
  frame"), so this module returns the ``(N, n_out)`` regression output and nothing else.

Two documented alternatives are available for later experiments and are **not** the default:

* ``stem='resnet'`` — the standard 7x7 / stride-2 ImageNet stem, with the pretrained RGB
  kernel averaged over its colour channels to accept grayscale input.
* ``head='keras'`` — ``Linear(2048,1000) + ReLU + Dropout(0.2) + Linear(1000,n_out)``, the
  head used by the author's 2023 Keras port (verified by reading the ``model_config``
  attribute of ``legacy/TVnet/1st.h5`` and ``2nd.h5``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

try:  # torchvision is a hard requirement of this module
    from torchvision.models import ResNet50_Weights, resnet50
except Exception as _exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "tvnet.model requires torchvision (for the ImageNet ResNet-50 backbone)."
    ) from _exc

__all__ = ["TVNetResNet50", "build_model", "count_parameters"]


_STEMS = ("matlab", "resnet")
_HEADS = ("matlab", "keras")


def _load_pretrained_resnet50(pretrained: bool) -> nn.Module:
    """Return a torchvision ResNet-50, optionally with IMAGENET1K_V1 weights.

    Raises a clear error if the weights cannot be obtained, so that a failed download can
    never silently degrade into training from scratch (which would quietly cost accuracy).
    """
    if not pretrained:
        return resnet50(weights=None)
    try:
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain torchvision ResNet50_Weights.IMAGENET1K_V1 "
            f"({type(exc).__name__}: {exc}). The MATLAB network in "
            "network_code/ResNet50_chs.m starts from ImageNet weights, so training from "
            "scratch would not reproduce the paper. Fix network access or pre-populate the "
            "torch hub cache (see torch.hub.get_dir()), or pass pretrained=False "
            "explicitly if you really want random initialisation."
        ) from exc


def _grayscale_stem_weight(rgb_weight: torch.Tensor, in_ch: int) -> torch.Tensor:
    """Collapse a pretrained (64, 3, 7, 7) stem kernel down to ``in_ch`` input channels."""
    if in_ch == 3:
        return rgb_weight.clone()
    mean = rgb_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
    if in_ch == 1:
        return mean
    # DEVIATION: no MATLAB counterpart. For in_ch not in {1, 3} we tile the RGB-averaged
    # kernel and divide by in_ch so that a channel-replicated input yields the same response
    # as the original RGB stem. Only reachable via the non-default stem='resnet' branch.
    return mean.repeat(1, in_ch, 1, 1) / float(in_ch)


def _to_resnet_v1(backbone: nn.Module) -> None:
    """Move the stride from the 3x3 back onto the 1x1 in each stage's first block.

    torchvision ships ResNet-50 **v1.5** (stride on ``conv2``, the 3x3). MATLAB's imported
    ResNet-50 is the original **v1** layout: the shipped TVnet networks have
    ``res{3,4,5}a_branch2a`` as 1x1 convolutions with ``Stride [2, 2]`` while their 3x3
    ``branch2b`` is stride 1 (SPEC.md section 12). Convolution *shapes* are identical
    between the two, so ImageNet weights still load; only the stride attributes move.

    Mutates ``backbone`` in place. Idempotent.
    """
    for layer in (backbone.layer2, backbone.layer3, backbone.layer4):
        block = layer[0]
        if block.conv2.stride != (1, 1):
            block.conv1.stride = block.conv2.stride
            block.conv2.stride = (1, 1)


def _set_bn_eps(module: nn.Module, eps: float) -> None:
    """MATLAB's batchNormalizationLayer defaults to Epsilon = 1e-3, torchvision to 1e-5."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = eps


class TVNetResNet50(nn.Module):
    """ResNet-50 coordinate regressor, mirroring ``ResNet50_chs.m``.

    Parameters
    ----------
    in_ch : int
        Input channels. MATLAB uses ``imageInputLayer([row, col, chann])`` with
        ``chann = size(train_x, 3) == 1`` (grayscale cine frames).
    n_out : int
        Regression outputs. MATLAB uses ``pts = size(train_y, 2) == 4``, i.e.
        ``[row1, col1, row2, col2]`` in the network's own input frame.
    pretrained : bool
        Load ImageNet weights into ``bn1`` / ``layer1..layer4``. The new stem and both new
        linear layers are always randomly initialised (Xavier uniform, zero bias), matching
        MATLAB, where ``convolution2dLayer`` / ``fullyConnectedLayer`` are freshly created.
    stem : {'matlab', 'resnet'}
        ``'matlab'`` (default) = the paper's 2x2 / stride-1 / 'same' conv, keeping full
        input resolution. ``'resnet'`` = the standard pretrained 7x7 / stride-2 stem.
    head : {'matlab', 'keras'}
        ``'matlab'`` (default) = two bare affine layers. ``'keras'`` = the 2023 port's
        ``Linear + ReLU + Dropout(0.2) + Linear``.

    Notes
    -----
    Input is expected to be the per-image normalised ``(x - median) / iqr`` cine frame
    (SPEC.md section 5), **not** ImageNet-normalised RGB; that is what MATLAB feeds in too.
    """

    def __init__(
        self,
        in_ch: int = 1,
        n_out: int = 4,
        pretrained: bool = True,
        stem: str = "matlab",
        head: str = "matlab",
        variant: str = "v1",
        bn_eps: float = 1e-3,
        input_mean: float = 0.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if stem not in _STEMS:
            raise ValueError(f"stem must be one of {_STEMS}, got {stem!r}")
        if head not in _HEADS:
            raise ValueError(f"head must be one of {_HEADS}, got {head!r}")
        if variant not in ("v1", "v1.5"):
            raise ValueError(f"variant must be 'v1' or 'v1.5', got {variant!r}")
        if in_ch < 1:
            raise ValueError(f"in_ch must be >= 1, got {in_ch}")
        if n_out < 1:
            raise ValueError(f"n_out must be >= 1, got {n_out}")

        self.in_ch = int(in_ch)
        self.n_out = int(n_out)
        self.pretrained = bool(pretrained)
        self.stem_kind = stem
        self.head_kind = head
        self.variant = variant
        self.bn_eps = float(bn_eps)
        self.dropout = float(dropout)

        backbone = _load_pretrained_resnet50(pretrained)
        if variant == "v1":
            _to_resnet_v1(backbone)
        _set_bn_eps(backbone, self.bn_eps)

        # MATLAB's imageInputLayer normalises with 'zerocenter': trainNetwork records the
        # mean of the training data and the trained network subtracts it at predict time,
        # AFTER the (x - median)/iqr step. It is part of the model, not the data pipeline,
        # so it lives in a buffer and travels with the checkpoint. 0.0 makes it a no-op.
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))

        # --- stem -----------------------------------------------------------------
        if stem == "matlab":
            # MATLAB: convolution2dLayer(2, 64, 'Stride', 1, 'padding', 'same', 'Name','conv1')
            # kernel 2 with stride 1 needs a total padding of 1; MATLAB puts the odd extra
            # pixel on the bottom/right, and so does torch's padding='same' (verified: its
            # _reversed_padding_repeated_twice is [0, 1, 0, 1]).
            #
            # DEVIATION: bias=False. MATLAB's convolution2dLayer carries a bias term, but the
            # very next layer is bn_conv1, whose per-channel shift subsumes any constant the
            # conv bias could add; the two graphs are therefore equivalent up to a
            # reparameterisation, and dropping the bias is the standard conv->BN idiom (it is
            # also what torchvision's own conv1 does).
            #
            # ...but bias=True is used anyway. The shipped MATLAB networks DO carry a
            # non-zero conv1 bias (SPEC.md section 12), and their bn_conv1 statistics were
            # estimated with it present, so tvnet.matlab_net needs the slot to exist in
            # order to host the original weights faithfully. Keeping it costs 64 parameters
            # and changes nothing when training from scratch.
            self.stem = nn.Conv2d(
                self.in_ch, 64, kernel_size=2, stride=1, padding="same", bias=True
            )
        else:
            # Alternative for later experiments: keep the real ResNet-50 stem (7x7, stride 2).
            self.stem = nn.Conv2d(
                self.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # --- pretrained trunk: MATLAB layers 3..174 (bn_conv1 ... avg_pool) --------
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d(1) == MATLAB 'avg_pool'

        # --- new head: MATLAB fc1000 -> fc6 -> regressionLayer --------------------
        fc1 = nn.Linear(2048, 1000)
        fc2 = nn.Linear(1000, self.n_out)
        if head == "matlab":
            # fullyConnectedLayer is affine only: no ReLU, no dropout.
            self.head = nn.Sequential(fc1, fc2)
        else:
            # The ReLU is the substantive fix: without it the two Linears collapse to one
            # affine map. Dropout is separable from that
            # and is off by default here, because regularising a 4-output regression head
            # mainly slows convergence.
            layers = [fc1, nn.ReLU(inplace=True)]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(fc2)
            self.head = nn.Sequential(*layers)

        # --- initialise everything MATLAB creates fresh ---------------------------
        self._init_new_layers(backbone)

    # fc1/fc2 are views into self.head, not separate submodules, so that state_dict()
    # holds each parameter under exactly one name.
    @property
    def fc1(self) -> nn.Linear:
        """MATLAB ``fc1000``: the new ``fullyConnectedLayer(1000)``."""
        return self.head[0]

    @property
    def fc2(self) -> nn.Linear:
        """MATLAB ``fc6``: the new ``fullyConnectedLayer(pts)``."""
        return self.head[-1]

    # ------------------------------------------------------------------ init ----
    def _init_new_layers(self, backbone: nn.Module) -> None:
        """Xavier-uniform / zero-bias init for the layers MATLAB adds from scratch."""
        if self.stem_kind == "resnet" and self.pretrained:
            # Reuse the pretrained 7x7 kernel, collapsed to in_ch channels.
            with torch.no_grad():
                self.stem.weight.copy_(
                    _grayscale_stem_weight(backbone.conv1.weight.detach(), self.in_ch)
                )
        else:
            nn.init.xavier_uniform_(self.stem.weight)
        if self.stem.bias is not None:
            nn.init.zeros_(self.stem.bias)

        for linear in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    # --------------------------------------------------------------- forward ----
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Everything up to and including global average pooling -> ``(N, 2048)``."""
        # MATLAB imageInputLayer 'zerocenter' normalisation; a no-op when input_mean == 0.
        x = x - self.input_mean
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, in_ch, H, W)`` -> ``(N, n_out)`` raw pixel coordinates.

        No output activation and no target scaling: MATLAB's ``regressionLayer`` takes the
        raw ``train_y`` pixel coordinates, and the caller applies plain MSE.
        """
        if x.dim() != 4:
            raise ValueError(f"expected a 4-D (N, C, H, W) tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_ch:
            raise ValueError(f"expected {self.in_ch} input channel(s), got {x.shape[1]}")
        return self.head(self.forward_features(x))

    def extra_repr(self) -> str:
        return (
            f"in_ch={self.in_ch}, n_out={self.n_out}, pretrained={self.pretrained}, "
            f"stem={self.stem_kind!r}, head={self.head_kind!r}"
        )


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a dict-like or attribute-style config, else return ``default``.

    A ``'model'`` sub-section, if present, takes precedence over the top level.
    """
    if cfg is None:
        return default
    for scope in (_cfg_sub(cfg, "model"), cfg):
        if scope is None:
            continue
        if isinstance(scope, Mapping):
            if key in scope:
                return scope[key]
        elif hasattr(scope, key):
            return getattr(scope, key)
    return default


def _cfg_sub(cfg: Any, name: str) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name)
    return getattr(cfg, name, None)


def build_model(cfg: Any = None) -> TVNetResNet50:
    """Construct a :class:`TVNetResNet50` from a config object.

    ``cfg`` may be ``None``, a mapping, or any attribute-style object (argparse Namespace,
    dataclass, SimpleNamespace...). Recognised keys, all optional, read from ``cfg['model']``
    first and then from ``cfg`` itself: ``in_ch``, ``n_out``, ``pretrained``, ``stem``,
    ``head``, ``variant``, ``bn_eps``, ``input_mean``. Defaults reproduce
    ``ResNet50_chs.m`` exactly.

    Every constructor argument is forwarded, so a config that sets ``variant``,
    ``bn_eps`` or ``input_mean`` cannot be silently ignored (all three change the
    numbers, per SPEC.md section 12).
    """
    return TVNetResNet50(
        in_ch=int(_cfg_get(cfg, "in_ch", 1)),
        n_out=int(_cfg_get(cfg, "n_out", 4)),
        pretrained=bool(_cfg_get(cfg, "pretrained", True)),
        stem=str(_cfg_get(cfg, "stem", "matlab")),
        head=str(_cfg_get(cfg, "head", "matlab")),
        variant=str(_cfg_get(cfg, "variant", "v1")),
        bn_eps=float(_cfg_get(cfg, "bn_eps", 1e-3)),
        input_mean=float(_cfg_get(cfg, "input_mean", 0.0)),
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters in ``model`` (trainable ones by default)."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return int(sum(p.numel() for p in params))

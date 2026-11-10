"""Updater: turns gradients into parameter updates. Runs on every rank after the gradients are averaged.

Starting program = the training script's own optimizer, unchanged:

optimizer 0: torch.optim.adamw.AdamW
  group 0 (6 tensors: 0.weight, 0.bias, 2.weight, 2.bias, 4.weight, 4.bias)
    {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.01, "amsgrad": false, "maximize": false, "foreach": null, "capturable": false, "differentiable": false, "fused": null, "decoupled_weight_decay": true, "initial_lr": 0.001}
scheduler 0: torch.optim.lr_scheduler.CosineAnnealingLR {"T_max": 2340, "eta_min": 0.0, "base_lrs": [0.001], "last_epoch": 0, "_step_count": 1, "_is_initial": false, "_last_lr": [0.001]}
  stepped whenever the training script calls scheduler.step(), while follow_user_scheduler is True

You may replace any part of it.
"""

import torch

# Gradient clipping by global norm before the update, moved here from the training script with
# wrap(clip_grad_norm=...); None: no clipping in this program.
CLIP_GRAD_NORM = None


class Updater:
    # True: the training script's scheduler.step() calls still step the user's schedulers (as in the plain run).
    # A program that sets its own learning rates should leave this False.
    follow_user_scheduler = True

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opts = spec["user_optimizers"]
        self.scheds = spec["user_schedulers"]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self, grads, update):
        # grads[name] is the same tensor as param.grad (None if the parameter got no gradient this update).
        if CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_([p for _, p in self.named], CLIP_GRAD_NORM)
        for o in self.opts:
            o.step()

    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts], scheds=[s.state_dict() for s in self.scheds])

    def load_state_dict(self, state):
        for o, s in zip(self.opts, state["opts"]):
            o.load_state_dict(s)
        for o, s in zip(self.scheds, state["scheds"]):
            o.load_state_dict(s)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)

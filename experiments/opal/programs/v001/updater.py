"""Updater: AdamW with re-heated cosine LR schedule, carrying over momentum state from the previous optimizer."""

import math
import copy
import torch


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.total_updates = spec["total_updates"]
        self.start_update = spec["update"]
        self.lr_start = 0.001
        self.lr_end = 0.0
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 0.01

        params = [p for _, p in named_params]
        self.opt = torch.optim.AdamW(
            params, lr=self.lr_start, betas=self.betas,
            eps=self.eps, weight_decay=self.weight_decay
        )

        # Carry over state from the previous updater's optimizer
        if previous is not None:
            prev_opts = previous.opts
            if prev_opts:
                state = copy.deepcopy(prev_opts[0].state_dict())
                self.opt.load_state_dict(state)

        self.opts = [self.opt]

    @property
    def param_groups(self):
        return self.opt.param_groups

    def lr_at(self, update):
        span = self.total_updates - self.start_update
        if span <= 0:
            return self.lr_end
        t = (update - self.start_update) / span
        t = min(max(t, 0.0), 1.0)
        return self.lr_end + 0.5 * (self.lr_start - self.lr_end) * (1.0 + math.cos(math.pi * t))

    def step(self, grads, update):
        lr = self.lr_at(update)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()

    def state_dict(self):
        return {
            "opt": self.opt.state_dict(),
            "start_update": self.start_update,
            "total_updates": self.total_updates,
            "lr_start": self.lr_start,
            "lr_end": self.lr_end,
        }

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])
        self.start_update = state["start_update"]
        self.total_updates = state["total_updates"]
        self.lr_start = state["lr_start"]
        self.lr_end = state["lr_end"]


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)

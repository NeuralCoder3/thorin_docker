# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from modules.PyTorch.utils import to_torch_tensors, torch_jacobian
from shared.ITest import ITest
from shared.GMMData import GMMInput, GMMOutput
from modules.PyTorch.gmm_objective3 import gmm_objective
import torch._dynamo
import torch._dynamo.config
import logging

# gmm_objective = torch.compile(gmm_objective, backend="eager")
# gmm_objective = torch.compile(gmm_objective, backend="aot_eager")
# torch._dynamo.config.log_level = logging.INFO
gmm_objective = torch.compile(gmm_objective, backend="inductor")
# gmm_objective = torch.compile(gmm_objective, backend="inductor", mode="reduce-overhead", fullgraph=True)
# gmm_objective = torch.compile(gmm_objective, backend="inductor", mode="reduce-overhead", fullgraph=False)
# gmm_objective = torch.compile(gmm_objective, backend="inductor", dynamic=True)

# mode="reduce-overhead"


class PyTorchGMM3(ITest):
    '''Test class for GMM differentiation by PyTorch.'''

    def prepare(self, input):
        '''Prepares calculating. This function must be run before
        any others.'''

        self.inputs = to_torch_tensors(
            (input.alphas, input.means, input.icf),
            grad_req=True
        )

        self.params = to_torch_tensors(
            (input.x, input.wishart.gamma, input.wishart.m)
        )

        self.objective = torch.zeros(1)
        self.gradient = torch.empty(0)

        print("Starting torch.compile()...")

        # torch._dynamo.explain(gmm_objective, *self.inputs, *self.params)
        _ = gmm_objective(*self.inputs, *self.params)

        print("PyTorchGMM2.prepare() finished")

    def output(self):
        '''Returns calculation result.'''

        return GMMOutput(self.objective.item(), self.gradient.numpy())

    # @torch.compile
    def calculate_objective(self, times):
        '''Calculates objective function many times.'''

        for i in range(times):
            self.objective = gmm_objective(*self.inputs, *self.params)

    # @torch.compile
    def calculate_jacobian(self, times):
        '''Calculates objective function jacobian many times.'''

        for i in range(times):
            self.objective, self.gradient = torch_jacobian(
                gmm_objective,
                self.inputs,
                self.params
            )

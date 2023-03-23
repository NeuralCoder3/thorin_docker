# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from modules.PyTorch.utils import to_torch_tensors, torch_jacobian
from shared.ITest import ITest
from shared.GMMData import GMMInput, GMMOutput
from modules.PyTorch.gmm_objective2 import gmm_objective

torch.set_num_threads(1)


class PyTorchGMM2S(ITest):
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

        #  torch.compile(model, backend="inductor")
        # self.calculate_objective = torch.compile(
        #     self.calculate_objective, backend="inductor")
        # self.calculate_jacobian = torch.compile(
        #     self.calculate_jacobian, backend="inductor")

        # https://stackoverflow.com/questions/394770/override-a-method-at-instance-level
        _ = gmm_objective(*self.inputs, *self.params)

        print("PyTorchGMM2.prepare() finished")

        # self.calculate_objective(1)
        # self.calculate_jacobian(1)

        # self.objective = torch.zeros(1)
        # self.gradient = torch.empty(0)

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

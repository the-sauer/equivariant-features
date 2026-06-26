# Copyright 2019 EPFL, Google LLC
# Copyright 2025-2026 Hendrik Sauer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class L2Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1, keepdims=True) + self.eps)
        norm = torch.where(norm > self.eps, norm, torch.tensor(1.0, device=x.device))
        x = x / norm
        return x

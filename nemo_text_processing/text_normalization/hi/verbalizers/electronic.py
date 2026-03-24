# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, NEMO_NOT_QUOTE

class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic data, e.g.
        electronic { value: "एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश एन ए ए एम डॉट कॉम" } -> https://naam.com
        electronic { value: "एच ई एल एल ओ एक दो तीन ऐट ओ यू टी एल ओ ओ के डॉट कॉम" } -> hello123@outlook.com
        electronic { value: "ए एल आई आई हाइफ़न पी ए ए टी टी आई एल डॉट कॉम" } -> alii-paattil.com
        electronic { value: "ए ए एस सी डॉट एन आई सी डॉट इन फॉरवर्ड स्लैश" } -> aasc.nic.in/

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic=True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        value = pynutil.delete('value: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        self.fst = self.delete_tokens(value).optimize()

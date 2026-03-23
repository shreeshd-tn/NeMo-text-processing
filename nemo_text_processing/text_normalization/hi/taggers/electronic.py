# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

class ElectronicFst(GraphFst):
    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        letter = pynini.string_file(get_abs_path("data/electronic/letters.tsv")) + insert_space
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) + insert_space
        symbol = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")) + insert_space
        dot = pynini.cross(".", "डॉट") + insert_space
        at = pynini.cross("@", "ऐट") + insert_space

        alnum = letter | digit
        token = alnum | symbol

        domain_word = (
            pynutil.add_weight(
                pynini.string_file(get_abs_path("data/electronic/domain.tsv")) + insert_space, -1.0
            )
            | pynini.closure(alnum, 0) + letter + pynini.closure(alnum, 0)
        )

        domain = domain_word + pynini.closure(dot + domain_word, 1)

        username = pynini.closure(token | dot, 1)

        email = username + at + domain

        proto_char = letter | digit
        protocol_str = pynini.closure(proto_char, 1)
        path = pynini.closure(token | dot | symbol, 1)
        url = protocol_str + pynini.closure(symbol, 1) + domain + pynini.closure(path, 0, 1)

        plain_domain = domain + pynini.closure(symbol, 0)

        graph = (email | url | plain_domain).optimize()

        graph = (
            pynutil.insert('value: "')
            + graph
            + pynutil.insert('"')
        )

        self.fst = self.add_tokens(graph)

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

        letters = pynini.string_file(get_abs_path("data/electronic/letters.tsv")) + insert_space
        digits = pynini.string_file(get_abs_path("data/electronic/digits.tsv")) + insert_space
        symbols = insert_space + pynini.string_file(get_abs_path("data/electronic/symbols.tsv")) + insert_space
        domains = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))
        protocols = pynini.string_file(get_abs_path("data/electronic/protocol.tsv"))

        token = letters | digits | symbols

        name = pynini.closure(domains | token, 1)

        dot = insert_space + pynini.cross(".", "डॉट") + insert_space

        domain = name + pynini.closure(dot + (domains | name), 1)

        email = name + pynini.cross("@", "ऐट ") + domain

        path = pynini.closure(symbols | letters | digits, 1)

        url = (
            pynini.closure(protocols + insert_space, 0, 1)
            + domain
            + pynini.closure(path, 0)
        )

        graph = email | url | domain

        graph = pynutil.insert('value: "') + graph + pynutil.insert('"')

        self.fst = self.add_tokens(graph.optimize())

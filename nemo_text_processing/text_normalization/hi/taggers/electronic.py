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
from nemo_text_processing.text_normalization.hi.graph_utils import (
    GraphFst,
    insert_space,
    TO_UPPER,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic addresses (URLs, emails, domains).
    e.g. laal.com -> electronic { value: "एल ए ए एल डॉट कॉम" }
         hello123@outlook.com -> electronic { value: "एच ई एल एल ओ एक दो तीन ऐट ओ यू टी एल ओ ओ के डॉट कॉम" }
         https://naam.com/faq/ -> electronic { value: "एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश एन ए ए एम डॉट कॉम फॉरवर्ड स्लैश एफ ए क्यू फॉरवर्ड स्लैश" }
         alii-paattil.com -> electronic { value: "ए एल आई आई हाइफ़न पी ए ए टी टी आई एल डॉट कॉम" }
         aasc.nic.in/ -> electronic { value: "ए ए एस सी डॉट एन आई सी डॉट इन फॉरवर्ड स्लैश" }
    """
    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        letter_base = pynini.string_file(get_abs_path("data/address/letters.tsv"))
        letter = (letter_base | pynini.compose(TO_UPPER, letter_base)) + insert_space
        digit = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        ) + insert_space
        dot = pynini.cross(".", "डॉट") + insert_space
        at = pynini.cross("@", "ऐट") + insert_space
        slash = pynini.cross("/", "फॉरवर्ड स्लैश") + insert_space
        hyphen = pynini.cross("-", "हाइफ़न") + insert_space
        underscore = pynini.cross("_", "अंडरस्कोर") + insert_space
        protocol = pynini.string_file(get_abs_path("data/electronic/protocol.tsv")) + insert_space
        protocol_sep = pynini.cross("://", "कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश") + insert_space

        alnum = letter | digit

        domain_word = (
            pynutil.add_weight(
                pynini.string_file(get_abs_path("data/electronic/domain.tsv")) + insert_space,
                -1.0,
            )
            | pynini.closure(digit, 0) + letter + pynini.closure(alnum | hyphen, 0) + (letter | digit)
        )
        domain = domain_word + pynini.closure(dot + domain_word, 1)

        octet = pynini.closure(digit, 1, 3)
        ip_address = octet + dot + octet + dot + octet + dot + octet

        username = pynini.closure(alnum | dot | hyphen | underscore, 1)
        email = username + at + domain

        segment = pynini.closure(alnum | hyphen | underscore | dot, 1)
        path = pynini.closure(slash + segment, 0) + pynini.closure(slash, 0, 1)
        url = protocol + protocol_sep + domain + path

        plain_domain = domain + pynini.closure(slash, 0, 1)

        graph = (email | url | ip_address | plain_domain).optimize()
        graph = (
            pynutil.insert('value: "')
            + graph
            + pynutil.insert('"')
        )
        self.fst = self.add_tokens(graph).optimize()

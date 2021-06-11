#
# Copyright (c) 2019-2021 James Thorne.
#
# This file is part of factual error correction.
# See https://jamesthorne.co.uk for further info.
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
#
import json
import logging
import os

logger = logging.getLogger(__name__)


class Reader:
    def __init__(self, labels, test=False):
        self.labels = labels
        self.test = test

    def read(self, path):
        logger.info("reading instances from {}".format(path))

        with open(path) as f:
            for idx, line in enumerate(f):
                instance = json.loads(line)
                yield from self.generate_instances(instance)

                if os.getenv("DEBUG") is not None and idx > 10:
                    break

    def generate_instances(self, instance):
        raise NotImplementedError()

    def maybe_format(self, page, evidence):
        return f"title: {self.clean(page)} context: {self.clean(evidence)}"

    @staticmethod
    def deduplicate(evidence):
        unique = set(map(lambda ev: (ev["page"], ev["line"]), evidence))
        return unique

    @staticmethod
    def clean(page):
        return (
            page.replace("_", " ")
            .replace("-LRB-", "(")
            .replace("-RRB-", ")")
            .replace("-COLON-", ":")
        )

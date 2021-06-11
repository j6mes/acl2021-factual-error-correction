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
from error_correction.modelling.reader.fever_database import FEVERDocumentDatabase
from error_correction.modelling.reader.reader import Reader


class SupervisedCorrectionReader(Reader):
    def __init__(self, labels, test=False):
        super().__init__(labels, test)
        self.db = FEVERDocumentDatabase("resources/wikipedia/fever.db")
        self.using_gold = False
        self.using_pipeline = False

    def generate_instances(self, instance):
        if instance["verdict"] is None or (
            instance["verdict"] not in self.labels and not self.test
        ):
            return None

        collected_evidence = []
        if "pipeline_text" in instance:
            assert not self.using_gold
            self.using_pipeline = True

            for page, evidence in instance["pipeline_text"]:
                collected_evidence.append(self.maybe_format(page, evidence))
        else:
            assert not self.using_pipeline
            self.using_gold = True

            for page, line in self.deduplicate(instance["evidence"]):

                if page is None or line is None:
                    continue

                found_page = self.db.get_doc_lines(page.split("#")[0])
                if found_page is None:
                    print("Page {} not found".format(page))
                    continue

                found_page = found_page.split("\n")
                assert line < len(found_page)

                ev_splits = found_page[line].split("\t")
                assert len(ev_splits) > 0

                evidence = found_page[line].split("\t")[1].strip()
                if len(evidence) == 0:
                    print("Zero evidence for: {} {}".format(page, line))
                    continue

                assert len(evidence) > 0

                collected_evidence.append(self.maybe_format(page, evidence))

        evidence = " ### ".join(collected_evidence)
        a = {
            "source": instance["mutated"],
            "target": instance["original"],
            "evidence": evidence,
            "mutation_type": instance["mutation"],
            "veracity": instance["verdict"],
        }
        yield a

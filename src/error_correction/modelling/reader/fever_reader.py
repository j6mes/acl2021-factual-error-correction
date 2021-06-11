import random

from error_correction.modelling.reader.fever_database import FEVERDocumentDatabase
from error_correction.modelling.reader.reader import Reader


class FEVERReader(Reader):
    def __init__(self, test):
        super().__init__(None, test)
        self.db = FEVERDocumentDatabase("resources/wikipedia/fever.db")
        self.using_pipeline = False
        self.using_gold = False

    def generate_instances(self, instance):
        collected_evidence = []
        if "pipeline_text" in instance:
            assert not self.using_gold
            self.using_pipeline = True

            for page, evidence in instance["pipeline_text"]:
                collected_evidence.append(self.maybe_format(page, evidence))
        else:
            assert not self.using_pipeline
            self.using_gold = True

            evs = set()

            for e in instance["evidence"]:
                evs.update([(ev[2], ev[3]) for ev in e])

            for page, line in evs:
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
            "source": instance["claim"],
            "evidence": evidence,
            "label": instance["label"],
            "metadata": instance,
        }

        yield a


class FEVERReaderSubSample(Reader):
    def __init__(self, test):
        super().__init__(None, test)
        self.db = FEVERDocumentDatabase("resources/wikipedia/fever.db")
        self.using_pipeline = False
        self.using_gold = False
        self.random = random.Random(1)

    def generate_instances(self, instance):

        if instance["label"] == "SUPPORTS" and not self.test:
            if self.random.uniform(0, 1) > 0.5:
                return

        collected_evidence = []
        if "pipeline_text" in instance:
            assert not self.using_gold
            self.using_pipeline = True

            for page, evidence in instance["pipeline_text"]:
                collected_evidence.append(self.maybe_format(page, evidence))
        else:
            assert not self.using_pipeline
            self.using_gold = True

            evs = set()

            for e in instance["evidence"]:
                evs.update([(ev[2], ev[3]) for ev in e])

            for page, line in evs:
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
            "source": instance["claim"],
            "evidence": evidence,
            "label": instance["label"],
        }

        yield a


class FEVERReaderNonNeutral(Reader):
    def __init__(self, test):
        super().__init__(None, test)
        self.db = FEVERDocumentDatabase("resources/wikipedia/fever.db")
        self.using_pipeline = False
        self.using_gold = False
        self.random = random.Random(1)

    def generate_instances(self, instance):

        if instance["label"] == "NOT ENOUGH INFO" and not self.test:
            return

        collected_evidence = []
        if "pipeline_text" in instance:
            assert not self.using_gold
            self.using_pipeline = True

            for page, evidence in instance["pipeline_text"]:
                collected_evidence.append(self.maybe_format(page, evidence))
        else:
            assert not self.using_pipeline
            self.using_gold = True

            evs = set()

            for e in instance["evidence"]:
                evs.update([(ev[2], ev[3]) for ev in e])

            for page, line in evs:
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
            "source": instance["claim"],
            "evidence": evidence,
            "label": instance["label"],
        }

        yield a

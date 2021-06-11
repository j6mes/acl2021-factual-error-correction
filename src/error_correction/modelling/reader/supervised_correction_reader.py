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

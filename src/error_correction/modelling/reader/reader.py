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

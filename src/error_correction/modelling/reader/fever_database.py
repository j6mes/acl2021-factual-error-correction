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
from functools import lru_cache
from drqa.retriever import DocDB, utils
import logging

logger = logging.getLogger(__name__)


class FEVERDocumentDatabase(DocDB):
    def __init__(self, path=None):
        super().__init__(path)
        logger.info(f"Use FEVER db: {path}")

    @lru_cache(maxsize=1000)
    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?", (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_doc_text(self, doc_id):
        lines = self.get_doc_lines(doc_id)

        if lines is None:
            return None

        lines = lines.split("\n")
        return "\n".join(
            [line.split("\t")[1] for line in lines if len(line.split("\t")) > 1]
        )

    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(lines)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results


class FEVERDocumentDatabaseIterable(FEVERDocumentDatabase):
    def iter_all_doc_lines(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id,lines FROM documents", ())
        result = cursor.fetchall()
        return result

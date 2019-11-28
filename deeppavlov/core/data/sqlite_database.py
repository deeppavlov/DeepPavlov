# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import sqlite3
from logging import getLogger
from typing import List, Dict

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator

log = getLogger(__name__)


@register('sqlite_database')
class Sqlite3Database(Estimator):
    """
    Loads and trains sqlite table of any items (with name ``table_name``
    and path ``save_path``).

    Primary (unique) keys must be specified, all other keys are infered from data.
    Batch here is a list of dictionaries, where each dictionary corresponds to an item.
    If an item doesn't contain values for all keys, then missing values will be stored
    with ``unknown_value``.

    Parameters:
        save_path: sqlite database path.
        primary_keys: list of table primary keys' names.
        keys: all table keys' names.
        table_name: name of the sqlite table.
        unknown_value: value assigned to missing item values.
        **kwargs: parameters passed to parent
            :class:`~deeppavlov.core.models.estimator.Estimator` class.
    """

    def __init__(self,
                 save_path: str,
                 primary_keys: List[str],
                 keys: List[str] = None,
                 table_name: str = "mytable",
                 unknown_value: str = 'UNK',
                 *args, **kwargs) -> None:
        super().__init__(save_path=save_path, *args, **kwargs)

        self.primary_keys = primary_keys
        if not self.primary_keys:
            raise ValueError("Primary keys list can't be empty")
        self.tname = table_name
        self.keys = keys
        self.unknown_value = unknown_value

        self.conn = sqlite3.connect(str(self.save_path),
                                    check_same_thread=False)
        self.cursor = self.conn.cursor()
        if self._check_if_table_exists():
            log.info(f"Loading database from {self.save_path}.")
            if not self.keys:
                self.keys = self._get_keys()
        else:
            log.info(f"Initializing empty database on {self.save_path}.")

    def __call__(self, batch: List[Dict],
                 order_by: str = None,
                 ascending: bool = False) -> List[List[Dict]]:
        order = 'ASC' if ascending else 'DESC'
        if not self._check_if_table_exists():
            log.warning("Database is empty, call fit() before using.")
            return [[] for i in range(len(batch))]
        return [self._search(b, order_by=order_by, order=order) for b in batch]

    def _check_if_table_exists(self):
        self.cursor.execute(f"SELECT name FROM sqlite_master"
                            f" WHERE type='table'"
                            f" AND name='{self.tname}';")
        return bool(self.cursor.fetchall())

    def _search(self, kv=None, order_by=None, order=''):
        order_expr = f" ORDER BY {order_by} {order}" if order_by else ''
        if kv:
            keys, values = zip(*kv.items())
            where_expr = " AND ".join(f"{k}=?" for k in keys)
            self.cursor.execute(f"SELECT * FROM {self.tname} WHERE {where_expr}" + order_expr, values)
        else:
            self.cursor.execute(f"SELECT * FROM {self.tname}" + order_expr)
        return [self._wrap_selection(s) for s in self.cursor.fetchall()]

    def _wrap_selection(self, selection):
        if not self.keys:
            self.keys = self._get_keys()
        return {f: v for f, v in zip(self.keys, selection)}

    def _get_keys(self):
        self.cursor.execute(f"PRAGMA table_info({self.tname});")
        return [info[1] for info in self.cursor]

    def _get_types(self):
        self.cursor.execute(f"PRAGMA table_info({self.tname});")
        return {info[1]: info[2] for info in self.cursor}

    def fit(self, data: List[Dict]) -> None:
        if not self._check_if_table_exists():
            self.keys = self.keys or [key for key in data[0]]
            # because in the next line we assume that in the first dict there are all (!) necessary keys:
            types = ('integer' if isinstance(data[0][k], int) else 'text' for k in self.keys)
            self._create_table(self.keys, types)
        elif not self.keys:
            self.keys = self._get_keys()

        self._insert_many(data)

    def _create_table(self, keys, types):
        if any(pk not in keys for pk in self.primary_keys):
            raise ValueError(f"Primary keys must be from {keys}.")
        new_types = (f"{k} {t} primary key"
                     if k in self.primary_keys else f"{k} {t}"
                     for k, t in zip(keys, types))
        new_types_joined = ', '.join(new_types)
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.tname}"
                            f" ({new_types_joined})")
        log.info(f"Created table with keys {self._get_types()}.")

    def _insert_many(self, data):
        to_insert = {}
        to_update = {}
        for kv in filter(None, data):
            primary_values = tuple(kv[pk] for pk in self.primary_keys)
            record = tuple(kv.get(k, self.unknown_value) for k in self.keys)
            curr_record = self._get_record(primary_values)
            if curr_record:
                if primary_values in to_update:
                    curr_record = to_update[primary_values]
                if curr_record != record:
                    to_update[primary_values] = record
            else:
                to_insert[primary_values] = record

        if to_insert:
            fformat = ','.join(['?'] * len(self.keys))
            self.cursor.executemany(f"INSERT into {self.tname}" +
                                    f" VALUES ({fformat})",
                                    to_insert.values())
        if to_update:
            for record in to_update.values():
                self._update_one(record)

        self.conn.commit()

    def _get_record(self, primary_values):
        ffields = ", ".join(self.keys) or "*"
        where_expr = " AND ".join(f"{pk}=?" for pk in self.primary_keys)
        fetched = self.cursor.execute(f"SELECT {ffields} FROM {self.tname}" +
                                      f" WHERE {where_expr}", primary_values).fetchone()
        if not fetched:
            return None
        return fetched

    def _update_one(self, record):
        set_values, where_values = [], []
        set_fields, where_fields = [], []
        for k, v in zip(self.keys, record):
            if k in self.primary_keys:
                where_fields.append(f"{k}=?")
                where_values.append(v)
            else:
                set_fields.append(f"{k}=?")
                set_values.append(v)
        set_expr = ", ".join(set_fields)
        where_expr = " AND ".join(where_fields)
        self.cursor.execute(f"UPDATE {self.tname}" +
                            f" SET {set_expr}" +
                            f" WHERE {where_expr}", set_values+where_values)

    def save(self):
        pass

    def load(self):
        pass

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Sequence
import knime.extension as knext
from langchain_core.messages import HumanMessage
import pandas as pd
from ._common import render_structured


@dataclass
class Port:
    name: str
    description: str
    type: str
    spec: Optional[dict] = None


@dataclass
class DataItem:
    """Represents a data item in the registry."""

    id: str
    meta_data: Port
    data: knext.Table


class DataRegistry:
    def __init__(self, data_message_prefix: str):
        self._data: list[DataItem] = []
        self._data_message_prefix = data_message_prefix

    @classmethod
    def create_with_input_tables(
        cls,
        input_tables: Sequence[knext.Table],
        input_ids: list[str] = [],
        data_message_prefix: str = None,
    ) -> "DataRegistry":
        """Creates a DataRegistry with the given input tables and optional prefix."""
        registry = cls(data_message_prefix=data_message_prefix)
        for i, table in enumerate(input_tables):
            spec = _spec_representation(table)
            port = Port(
                name=f"input_table_{i + 1}",
                description=f"Input table {i + 1}",
                type="Table",
                spec=spec,
            )
            if i >= len(input_ids):
                id = str(i + 1)
            else:
                id = input_ids[i]
            registry._data.append(DataItem(id=id, meta_data=port, data=table))
        return registry

    @classmethod
    def load(cls, meta_data: dict, tables: list[knext.Table]) -> "DataRegistry":
        """Creates a data registry from the given dictionary and list of tables (as returned by dump())."""
        data_message_prefix = meta_data.get("data_message_prefix", "")

        registry = cls(data_message_prefix=data_message_prefix)

        # Restore the data items from ports metadata and actual data
        ports_metadata = meta_data["ports"]
        ids = meta_data["ids"]

        for i, port_dict in enumerate(ports_metadata):
            # Convert dictionary back to Port object
            port = Port(
                name=port_dict["name"],
                description=port_dict["description"],
                type=port_dict["type"],
                spec=port_dict.get("spec"),
            )

            # Get the corresponding table data
            table = tables[i] if i < len(tables) else None
            if table is not None:
                registry._data.append(DataItem(id=ids[i], meta_data=port, data=table))

        return registry

    def dump(self):
        """Dumps the state of current data registry into a dictionary and a list of tables."""
        data_registry_info = {
            "ids": [],
            "ports": [],
            "data_message_prefix": self._data_message_prefix,
        }
        for id, data in enumerate(self._data):
            data_registry_info["ports"].append(port_to_dict(data.meta_data))
            data_registry_info["ids"].append(data.id)
        tables = [self._data[i].data for i in range(len(self._data))]
        return data_registry_info, tables

    def add_table(self, id: str, table: knext.Table, port: Port) -> dict:
        spec = _spec_representation(table)
        meta_data = Port(
            name=port.name,
            description=port.description,
            type=port.type,
            spec=spec,
        )
        if id is None:
            id = str(len(self._data) + 1)
        self._data.append(DataItem(id=id, meta_data=meta_data, data=table))
        return {id: port_to_dict(meta_data)}

    def get_data(self, id: any) -> knext.Table:
        for data_item in self._data:
            if data_item.id == str(id):
                return data_item.data
        raise ValueError(
            f"No data item found with id: {id}. Available ids: {[data.id for data in self._data]}"
        )

    def get_last_tables(self, num_tables: int) -> list[knext.Table]:
        """Returns the last `num_tables` tables added to the registry."""
        if num_tables <= 0:
            return []
        tables = [data_item.data for data_item in self._data[-num_tables:]]
        if len(tables) < num_tables:
            empty_table = _empty_table()
            tables = tables + [empty_table] * (num_tables - len(tables))
        return tables

    @property
    def has_data(self) -> bool:
        """Returns True if there is data in the registry, False otherwise."""
        return len(self._data) > 0

    def create_data_message(self) -> HumanMessage:
        if self._data:
            content = render_structured(
                **{
                    data.id: port_to_dict(data.meta_data)
                    for id, data in enumerate(self._data)
                }
            )
        else:
            content = "No data available. Use a tool to produce data."

        return HumanMessage(self._data_message_prefix + content)


def _spec_representation(table: knext.Table) -> dict:
    return {"columns": {column.name: str(column.ktype) for column in table.schema}}


def port_to_dict(port: Port) -> dict:
    return {
        "name": port.name,
        "description": port.description,
        "type": port.type,
        "spec": port.spec,
    }


def _empty_table():
    """Returns an empty knext.Table."""
    # Assuming knext.Table() creates an empty table, adjust as necessary
    return knext.Table.from_pandas(pd.DataFrame())

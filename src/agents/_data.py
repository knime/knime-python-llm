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

    meta_data: Port
    data: knext.Table


class DataRegistry:
    def __init__(self, data_message_prefix: str):
        self._data: list[DataItem] = []
        self._data_message_prefix = data_message_prefix

    @classmethod
    def create_with_input_tables(
        cls, input_tables: Sequence[knext.Table], data_message_prefix: str = None
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
            registry._data.append(DataItem(meta_data=port, data=table))
        return registry

    @classmethod
    def create_from_view_data(cls, view_data: dict) -> "DataRegistry":
        """Creates a DataRegistry from the given view data dictionary."""
        data_registry_info = view_data["data"]["data_registry"]
        data_message_prefix = data_registry_info.get("data_message_prefix", "")

        registry = cls(data_message_prefix=data_message_prefix)

        # Restore the data items from ports metadata and actual data
        ports_metadata = data_registry_info["ports"]
        table_data = view_data["ports"]

        for i, port_dict in enumerate(ports_metadata):
            # Convert dictionary back to Port object
            port = Port(
                name=port_dict["name"],
                description=port_dict["description"],
                type=port_dict["type"],
                spec=port_dict.get("spec"),
            )

            # Get the corresponding table data
            table = table_data[i] if i < len(table_data) else None
            if table is not None:
                registry._data.append(DataItem(meta_data=port, data=table))

        return registry

    def dump_into_view_data(self, view_data: dict):
        """Dumps the current data registry into the given view data dictionary."""
        data_registry_info = {
            "ports": [],
            "data_message_prefix": self._data_message_prefix,
        }
        for id, data in enumerate(self._data):
            data_registry_info["ports"].append(port_to_dict(data.meta_data))
        view_data["data"]["data_registry"] = data_registry_info
        view_data["ports"] = [self._data[i].data for i in range(len(self._data))]

    def add_table(self, table: knext.Table, port: Port) -> dict:
        spec = _spec_representation(table)
        meta_data = Port(
            name=port.name, description=port.description, type=port.type, spec=spec
        )
        self._data.append(DataItem(meta_data=meta_data, data=table))
        return {str(len(self._data) - 1): port_to_dict(meta_data)}

    def get_data(self, index: int) -> knext.Table:
        if index < 0 or index >= len(self._data):
            raise IndexError("Index out of range")
        return self._data[index].data

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
                    str(id): port_to_dict(data.meta_data)
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

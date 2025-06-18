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
    def create_with_input_tables(cls, input_tables: Sequence[knext.Table], data_message_prefix: str = None) -> "DataRegistry":
        """Creates a DataRegistry with the given input tables and optional prefix."""
        registry = cls(data_message_prefix=data_message_prefix)
        for i, table in enumerate(input_tables):
            spec = _spec_representation(table)
            port = Port(name=f"input_table_{i+1}", description=f"Input table {i+1}", type="Table", spec=spec)
            registry._data.append(DataItem(meta_data=port, data=table))
        return registry

    def add_table(self, table: knext.Table, port: Port) -> dict:
        spec = _spec_representation(table)
        meta_data = Port(
            name=port.name, description=port.description, type=port.type, spec=spec)
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
                **{str(id): port_to_dict(data.meta_data) for id, data in enumerate(self._data)}
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
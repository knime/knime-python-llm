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
    def __init__(self):
        self._data: list[DataItem] = []

    @classmethod
    def create_with_input_tables(self, input_tables: Sequence[knext.Table]) -> "DataRegistry":
        """Creates a DataRegistry with the given input tables."""
        registry = DataRegistry()
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
        return {len(self._data) - 1: port_to_dict(meta_data)}

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

    def create_data_message(self) -> Optional[HumanMessage]:
        msg = """# Data Tools Interface
You have access to tools that can consume and produce data.
The interaction with these tools is mediated via a data repository that keeps track of all available data items.
The repository is represented as a map from IDs to data items.

Each data item is represented by:
- The name of the data
- The description of the data
- The type of data
- The spec of the data giving a high-level overview of the data (e.g. the columns in a table)

Note: You do not have access to the actual data content, only the metadata and IDs.

# Using Tools with Data
## Consuming Data:
To pass data to a tool, provide the ID of the relevant data item.
Once invoked, the tool will receive the data associated with that ID.

## Producing Data:
- Tools that produce data will include an update to the data repository in their tool message.
- This update follows the same format as the initial data repository: A map of IDs to data items.

You must incorporate these updates into your working view of the data repository.
# Data:
"""
        if not self._data:
            return HumanMessage(msg + "No initial data available. Use a tool to produce data.")
        content = render_structured(**{str(id): port_to_dict(data.meta_data) for id, data in enumerate(self._data)})
        
        return HumanMessage(msg + content)

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
from typing import Any, Union

import sqlalchemy
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import Column, Table, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.sql.base import ReadOnlyColumnCollection

from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.chains.types.common import DataType, DataTypeWithUnit
from docugami_langchain.chains.types.data_type_detection_chain import (
    DataTypeDetectionChain,
)
from docugami_langchain.chains.types.date_parse_chain import DateParseChain
from docugami_langchain.chains.types.float_parse_chain import FloatParseChain
from docugami_langchain.config import BATCH_SIZE, TYPE_DETECTION_SAMPLE_SIZE
from docugami_langchain.output_parsers.truthy import TRUTHY_STRINGS


def _batch_process(chain: BaseDocugamiChain, inputs: list[str]) -> list[Any]:
    """Process inputs in batches using the chain's run_batch method."""
    results: list[Any] = []
    for i in range(0, len(inputs), BATCH_SIZE):
        batch = inputs[i : i + BATCH_SIZE]
        batch_results = chain.run_batch(inputs=batch)
        for result in batch_results:
            if isinstance(result, Exception):
                # ignore any items that failed to process in the batch
                results.append(None)
            else:
                results.append(result)

    return results


def _get_column_types(
    connection: Connection,
    columns: ReadOnlyColumnCollection[str, Column[Any]],
    data_type_detection_chain: DataTypeDetectionChain,
) -> dict[str, DataTypeWithUnit]:
    """Determine the predominant type for each TEXT column."""

    column_types: dict[str, DataTypeWithUnit] = {}

    for column in columns:
        if str(column.type).lower() != "text":
            continue  # Only process TEXT columns

        # Sample some rows for type detection
        sampling_select_query = select(column).limit(TYPE_DETECTION_SAMPLE_SIZE)
        sampling_result = connection.execute(sampling_select_query)

        sampled_col_values = [row[0] for row in sampling_result if row[0]]
        detected_types: list[Union[None, DataTypeWithUnit]] = _batch_process(
            data_type_detection_chain, sampled_col_values
        )

        type_counts: dict[DataTypeWithUnit, int] = {}
        for detected_type in detected_types:
            if detected_type:
                if detected_type not in type_counts:
                    type_counts[detected_type] = 0
                type_counts[detected_type] += 1

        # Determine the predominant type
        predominant_type = max(type_counts, key=lambda k: type_counts[k])
        column_types[column.name] = predominant_type

    return column_types


def _create_typed_table(
    connection: Connection,
    original_table: Table,
    column_types: dict[str, DataTypeWithUnit],
) -> str:
    """Create a new table with typed columns."""

    new_table_name = f"{original_table.name}_typed"
    create_table_stmt = f'CREATE TABLE "{new_table_name}" ('

    for column in original_table.columns:
        if column.name in column_types:
            dg_type = column_types[column.name]
            column_type = "TEXT"  # Default to TEXT if not identified

            if dg_type.type == DataType.NUMBER:
                column_type = "REAL"
            elif dg_type.type == DataType.DATETIME:
                column_type = "TEXT"  # Store as ISO8601 string
            elif dg_type.type == DataType.BOOL:
                column_type = "INTEGER"  # Store as 0 or 1

            # Include the unit in the column name if applicable
            new_column_name = column.name
            if dg_type.unit:
                new_column_name += f" ({dg_type.unit})"

            create_table_stmt += f'"{new_column_name}" {column_type}, '
        else:
            create_table_stmt += f'"{column.name}" TEXT, '

    create_table_stmt = create_table_stmt.rstrip(", ") + ")"
    connection.execute(text(create_table_stmt))

    return new_table_name


def _transfer_data_to_typed_table(
    connection: Connection,
    original_table: Table,
    typed_table: Table,
    column_types: dict[str, DataTypeWithUnit],
    date_parse_chain: DateParseChain,
    float_parse_chain: FloatParseChain,
) -> None:
    """Transfer data to the new typed table."""
    column_name_to_index = {
        col.name: idx for idx, col in enumerate(original_table.columns)
    }

    full_data_select_query = select(original_table)
    full_data_result = connection.execute(full_data_select_query)

    insert_stmt_prefix = f'INSERT INTO "{typed_table.name}" ('
    for column in typed_table.columns:
        insert_stmt_prefix += f'"{column.name}", '

    insert_stmt_prefix = insert_stmt_prefix.rstrip(", ") + ") VALUES "

    # Gather all rows to process
    all_rows = [row for row in full_data_result]

    # Prepare lists to batch process
    date_values = []
    float_values = []
    date_indices = []
    float_indices = []

    for row_idx, row in enumerate(all_rows):
        for column in original_table.columns:
            value = row[column_name_to_index[column.name]]
            if column.name in column_types and value:
                value_type = column_types[column.name].type
                if value_type == DataType.DATETIME:
                    date_values.append(value)
                    date_indices.append((row_idx, column.name))
                elif value_type == DataType.NUMBER:
                    float_values.append(value)
                    float_indices.append((row_idx, column.name))

    # Batch process date and float values
    date_results = _batch_process(date_parse_chain, date_values)
    float_results = _batch_process(float_parse_chain, float_values)

    # Create a dictionary to map the processed values back to their positions
    date_value_map = {
        date_indices[i]: (
            date_results[i].isoformat() if date_results[i] is not None else None
        )
        for i in range(len(date_indices))
    }
    float_value_map = {
        float_indices[i]: (float_results[i] if float_results[i] is not None else None)
        for i in range(len(float_indices))
    }

    # Insert processed values into the new table
    for row_idx, row in enumerate(all_rows):
        insert_stmt = insert_stmt_prefix + "("
        for column in original_table.columns:
            value = row[column_name_to_index[column.name]]
            if column.name in column_types:
                value_type = column_types[column.name].type
                converted_value = "NULL"  # default to NULL if not converted

                if (row_idx, column.name) in date_value_map:
                    if date_value_map[(row_idx, column.name)] is not None:
                        converted_value = f'"{date_value_map[(row_idx, column.name)]}"'  # store as quoted ISO str
                elif (row_idx, column.name) in float_value_map:
                    if float_value_map[(row_idx, column.name)] is not None:
                        converted_value = str(
                            float_value_map[(row_idx, column.name)]
                        )  # store as numeric
                elif value:
                    if value_type == DataType.TEXT:
                        converted_value = f'"{value}"'  # store as quoted text
                    elif value_type == DataType.BOOL:
                        if any(substring in value for substring in TRUTHY_STRINGS):
                            converted_value = "1"  # store as numeric 1
                        else:
                            converted_value = "0"  # store as numeric 0

                insert_stmt += f"{converted_value}, "
            else:
                insert_stmt += f'"{value}", '

        insert_stmt = insert_stmt.rstrip(", ") + ")"
        connection.execute(text(insert_stmt))


def convert_to_typed(
    db: SQLDatabase,
    data_type_detection_chain: DataTypeDetectionChain,
    date_parse_chain: DateParseChain,
    float_parse_chain: FloatParseChain,
) -> SQLDatabase:
    """
    Goes through all the tables in the database, and converts each TEXT column to a typed column where
    there is a predominant parseable data type detected.
    """
    inspector = sqlalchemy.inspect(db._engine)
    original_table_names = inspector.get_table_names()

    with db._engine.connect() as connection:
        with connection.begin():  # Ensure changes are committed
            for original_table_name in original_table_names:
                original_table = Table(
                    original_table_name, db._metadata, autoload_with=db._engine
                )

                # Get predominant types for each column
                column_types = _get_column_types(
                    connection, original_table.columns, data_type_detection_chain
                )

                # Create a new table with typed columns
                typed_table_name = _create_typed_table(
                    connection, original_table, column_types
                )
                typed_table = Table(
                    typed_table_name, db._metadata, autoload_with=db._engine
                )

                # Transfer data to the new typed table
                _transfer_data_to_typed_table(
                    connection,
                    original_table,
                    typed_table,
                    column_types,
                    date_parse_chain,
                    float_parse_chain,
                )

                # Drop the original table and rename the new one
                connection.execute(text(f'DROP TABLE "{original_table_name}"'))
                connection.execute(
                    text(
                        f'ALTER TABLE "{typed_table.name}" RENAME TO "{original_table_name}"'
                    )
                )

    # Close the existing connection
    db._engine.dispose()

    # Reconnect to refresh the database state
    refreshed_db = SQLDatabase.from_uri(
        str(db._engine.url), sample_rows_in_table_info=0
    )
    return refreshed_db
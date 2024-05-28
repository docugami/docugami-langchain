from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import Table, select, text
from sqlalchemy.engine.reflection import Inspector

from docugami_langchain.chains.types.common import DataTypes, DocugamiDataType
from docugami_langchain.chains.types.data_type_detection_chain import (
    DataTypeDetectionChain,
)
from docugami_langchain.chains.types.date_parse_chain import DateParseChain
from docugami_langchain.chains.types.float_parse_chain import FloatParseChain
from docugami_langchain.config import TYPE_DETECTION_SAMPLE_SIZE
from docugami_langchain.output_parsers.truthy import TRUTHY_STRINGS


def convert_to_typed(
    db: SQLDatabase,
    data_type_detection_chain: DataTypeDetectionChain,
    date_parse_chain: DateParseChain,
    float_parse_chain: FloatParseChain,
) -> None:
    """
    Goes through all the tables in the database, and converts each TEXT column to a typed column where
    there is a predominant parseable data type detected.
    """

    inspector = Inspector.from_engine(db._engine)
    table_names = inspector.get_table_names()

    with db._engine.connect() as connection:
        for table_name in table_names:
            table = Table(table_name, db._metadata, autoload_with=db._engine)
            columns = table.columns

            # Dictionary to store predominant types for each column
            column_types: dict[str, DocugamiDataType] = {}

            for column in columns:
                if str(column.type).lower() != "text":
                    continue  # Only process TEXT columns

                # Sample some rows for type detection
                sampling_select_query = select(column).limit(TYPE_DETECTION_SAMPLE_SIZE)
                sampling_result = connection.execute(sampling_select_query)

                type_counts = {
                    DataTypes.NUMBER: 0,
                    DataTypes.DATETIME: 0,
                    DataTypes.TEXT: 0,
                    DataTypes.BOOL: 0,
                }
                for row in sampling_result:
                    if row[0]:
                        detected_type = data_type_detection_chain.run(row[0])
                        type_counts[detected_type.value.type] += 1

                # Determine the predominant type
                predominant_type = max(type_counts, key=lambda k: type_counts[k])
                predominant_dtype = DocugamiDataType(type=predominant_type)
                column_types[column.name] = predominant_dtype

            # Create a new table with typed columns
            new_table_name = f"{table_name}_typed"
            create_table_stmt = f'CREATE TABLE "{new_table_name}" ('

            for column in columns:
                if column.name in column_types:
                    dg_type = column_types[column.name]
                    column_type = "TEXT"  # Default to TEXT if not identified

                    if dg_type.type == DataTypes.NUMBER:
                        column_type = "REAL"
                    elif dg_type.type == DataTypes.DATETIME:
                        column_type = "TEXT"  # Store as ISO8601 string
                    elif dg_type.type == DataTypes.BOOL:
                        column_type = "NUMBER"  # Store as 0 or 1

                    # Include the unit in the column name if applicable
                    new_column_name = column.name
                    if dg_type.unit:
                        new_column_name += f" ({dg_type.unit})"

                    create_table_stmt += f'"{new_column_name}" {column_type}, '
                else:
                    create_table_stmt += f'"{column.name}" TEXT, '

            create_table_stmt = create_table_stmt.rstrip(", ") + ")"
            connection.execute(text(create_table_stmt))

            # Create a column name to index mapping
            column_name_to_index = {col.name: idx for idx, col in enumerate(columns)}

            # Transfer data to the new typed table
            full_data_select_query = select(table)
            full_data_result = connection.execute(full_data_select_query)

            insert_stmt_prefix = f'INSERT INTO "{new_table_name}" ('
            for column in columns:
                insert_stmt_prefix += f'"{column.name}", '

            insert_stmt_prefix = insert_stmt_prefix.rstrip(", ") + ") VALUES "

            for row in full_data_result:
                insert_stmt = insert_stmt_prefix + "("
                for column in columns:
                    value = row[column_name_to_index[column.name]]

                    if column.name in column_types:
                        value_type = column_types[column.name].type
                        converted_value = "NULL"  # default to NULL if not converted

                        if value:
                            if value_type == DataTypes.TEXT:
                                converted_value = f'"{value}"'  # store as quoted text
                            elif value_type == DataTypes.BOOL:
                                if any(
                                    substring in value for substring in TRUTHY_STRINGS
                                ):
                                    converted_value = "1"  # store as numeric 1
                                else:
                                    converted_value = "0"  # store as numeric 0
                            elif value_type == DataTypes.DATETIME:
                                try:
                                    date_parse_response = date_parse_chain.run(value)
                                    converted_value = f'"{date_parse_response.value.isoformat()}"'  # store as quoted ISO str
                                except Exception:
                                    ...  # log?
                            elif value_type == DataTypes.NUMBER:
                                try:
                                    float_parse_response = float_parse_chain.run(value)
                                    converted_value = str(
                                        float_parse_response.value
                                    )  # store as numeric
                                except Exception:
                                    ...  # log?
                        insert_stmt += f"{converted_value}, "
                    else:
                        insert_stmt += f'"{value}", '

                insert_stmt = insert_stmt.rstrip(", ") + ")"
                connection.execute(text(insert_stmt))

            # Drop the original table and rename the new one
            connection.execute(text(f"DROP TABLE \"{table_name}\""))
            connection.execute(
                text(f"ALTER TABLE \"{new_table_name}\" RENAME TO \"{table_name}\"")
            )

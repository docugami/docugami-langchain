from langchain_core.output_parsers import BaseOutputParser


class SQLFindingOutputParser((BaseOutputParser[str])):
    """Parse the output of an LLM call to find a SQL statement inside it."""

    start_sequence: str = "SELECT"
    """string that marks the start of the SQL statement"""

    end_sequences: list[str] = [";", "\n\n", "```"]
    """strings that mark the end of SQL statements"""

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "sql_finding_output_parser"

    def parse(self, text: str) -> str:
        """Parse the output of an LLM call by extracting the entire first SQL statement
        from the given text.

        The start and end of the SQL  statement is detected by looking for one of the
        specified sequences

        Parameters:
        - text (str): The input string containing potential SQL statements.

        Returns:
        - str: The extracted SQL statement or an empty string if none found.

        Doctests:
        >>> parser = SQLFindingOutputParser()
        >>> parser.parse("random text SELECT * FROM users;")
        'SELECT * FROM users'

        >>> parser.parse("SELECT * FROM users WHERE name='John'; rest of the string")
        "SELECT * FROM users WHERE name='John'"

        >>> parser.parse("random text\\n\\nSELECT * FROM users\\nWHERE name='John'\\n"
        ... "\\nrest of the string")
        "SELECT * FROM users\\nWHERE name='John'"

        >>> parser.parse("random text\\n\\nSELECT * FROM users\\nWHERE name='John'```"
        ... "rest of the string")
        "SELECT * FROM users\\nWHERE name='John'"

        >>> parser.parse("SELECT * FROM \\"foo\\" WHERE \\"x\\" = 'y';\\n\\nSELECT * "
        ... "FROM \\"bar\\" WHERE \\"y\\" = 'z'")
        'SELECT * FROM \"foo\" WHERE \"x\" = \\'y\\''

        >>> parser.parse("SELECT * FROM \\"foo\\" WHERE \\"x\\" = 'y';\\n"
        ... "TABLE DESCRIPTION:")
        'SELECT * FROM \"foo\" WHERE \"x\" = \\'y\\''

        >>> parser.parse("No SQL statement here.")
        ''
        """

        # Find the starting point of the first statement
        start_idx = text.find(self.start_sequence)

        if start_idx == -1:
            return ""

        # Check each end sequence to find the nearest one
        end_idx = float("inf")
        for seq in self.end_sequences:
            seq_idx = text.find(seq, start_idx)
            if seq_idx != -1:
                end_idx = min(end_idx, seq_idx + len(seq))

        # If we didn't find an end sequence, return the entire string
        if end_idx == float("inf"):
            end_idx = len(text)

        sql_stmt = text[start_idx : int(end_idx)]
        for seq in self.end_sequences:
            sql_stmt = sql_stmt.rstrip(seq)

        return sql_stmt.strip()

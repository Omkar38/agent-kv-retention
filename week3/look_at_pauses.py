import duckdb

P = "our_work/pauses.parquet"

print("Columns:")
duckdb.sql(f"DESCRIBE SELECT * FROM '{P}'").show()

print("First 10 rows:")
duckdb.sql(f"SELECT * FROM '{P}' LIMIT 10").show()

print("The Claude session from step 2:")
duckdb.sql(f"""
    SELECT round_index, pause_s, cause, usable, n_tools, tool_names, context_tokens
    FROM '{P}'
    WHERE session_id = 'claude:9dda6cbb-174c-4536-b993-c79c836ac161'
    ORDER BY round_index
""").show()

# Optional: also save a CSV you can open in Excel. Remove the # to turn it on.
# duckdb.sql(f"COPY (SELECT * FROM '{P}') TO 'our_work/pauses.csv'")

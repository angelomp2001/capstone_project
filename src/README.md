Purpose: use a chatbot to clean your data and model a target column.

Components:
chat ui: user prompts df-edit task
llm: llm receives prompt + views function list -> outputs function.
parse: extract JSON from text
convert: convert JSON to function list
apply: apply to df

# data cleaning functions:
copy_df
validate required
validate unique
validate categorical values
validate bounds
validate date range
drop duplicates
replace value
coerce dtype
dropna
fillna
drop_columns


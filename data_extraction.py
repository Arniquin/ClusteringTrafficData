from tabula import read_pdf
import pandas as pd

# # Extract table from a single page
# df = read_pdf("./DataPDFs/dt79.pdf", pages=6, lattice=True)
# print(df)

def column_cleaning(tables):
    for table in tables:
        columns = table.columns
        for column in columns:
            if "\r" in column:
                col_temp = column
                col_temp = col_temp.replace("\r", " ")
                table.rename(columns={column: col_temp}, inplace=True)
    return tables   

def pdf_table_extraction(path, range, column_names = None, column_names_flag=False):
    if column_names_flag:
        tables = read_pdf(
        path,
        pages=range,
        lattice=True,
        pandas_options={
            "names": column_names
        }
    )
    else:
        tables = read_pdf(path, pages=range, lattice=True, multiple_tables=False, guess=True)
        tables = column_cleaning(tables) 
    return tables
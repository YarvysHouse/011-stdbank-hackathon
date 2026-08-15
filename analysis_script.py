from pathlib import Path 
import numpy as np 
import pandas as pd 

# -- 0. DATASOURCES 
BASE_DIR_ = Path(__file__).resolve().parent 
DATA_ = BASE_DIR_ / "Data" 
OUTPUTS_ = BASE_DIR_ / "Outputs" 


TRANSACTIONS_ = DATA_/"transactional_banking.csv"
CROSS_BORDER_ = DATA_/"cross_border_payments.csv"
TRADE_FINANCE_ = DATA_/"trade_finance.csv"
BENCHMARK_ = BASE_DIR_/"benchmarks"/"extraction_worklist_15_aug.csv"

PUBLIC_FINANCE_ = ""

# -- 1. LOAD DATA 
def load_data(file_path) -> pd.DataFrame:
    path = file_path
    df = pd.read_csv(path)

    # drop duplicate lines
    df = df.drop_duplicates()

    # Column Type Checking
        # DateTime
    df["date"] = pd.to_datetime(df["date"])
        # Column name change for transaction_df (consistency)
    if "amount_zar" in df:
        df = df.rename(columns={"amount_zar": "value_zar"})
    if "instrument_id" in df:
        df = df.rename(columns={"instrument_id":"transaction_id"})
        # Inbound-Outbound sign (negative)
    outflow = {"outbound", "import"}
    df["signed_amount"] = np.where(
            df["direction"].isin(outflow), -df["value_zar"], df["value_zar"]
    )
    month, year = df["date"].dt.month, df["date"].dt.year 
    df["fiscal_year"] = "FY"+(year + (month >= 7).astype(int)).astype(str)
        # 

    print("DF Shape: ", df.shape)
    print("DF Columns: ", df.columns)
    # Return DataFrame 
    return df.reset_index(drop=True)

def reference_type_check(df):
    df["reference"] = df["reference"].str[:-7]
    print(df["reference"].unique())
    return df

# -- 2 ENTITY ANALYSIS
def combined_df(df_1, df_2, df_3):
    # PARAMETERS (Inflows, Outflows)
    all_dfs = [df_1, df_2, df_3]

    cols = ["entity_id", "entity_name", "sector", "value_zar", "signed_amount", "transaction_id", "date", "fiscal_year", "reference"]

    

    df_consolidated = pd.concat(
        [ 
            df_1.assign(source="transactional")[cols + ["source"]],
            df_2.assign(source="cross_border")[cols + ["source"]],
            df_3.assign(source="trade_finance")[cols + ["source"]],
        ],
        ignore_index=True,
    )

    return df_consolidated

def entity_report(cons_df):
    df = cons_df.copy()
    df_b = cons_df.copy()
    # Aggregate
    df["inflow"] = df["signed_amount"].clip(lower=0)
    df["outflow"] = (-df["signed_amount"]).clip(lower=0)

    report_1 = df.groupby(["entity_id", "entity_name", "sector"], as_index=False).agg(
        num_transactions=("transaction_id", "nunique"),
        incomes=("inflow", "sum"),
        num_incomes=("inflow", lambda s: (s > 0).sum()),
        payments=("outflow", "sum"),
        num_payments=("outflow", lambda s: (s > 0).sum())
    )

    report_2 = df_b.pivot_table(
            index=["entity_id", "entity_name", "sector"],
            columns="reference",
            values="signed_amount",
            aggfunc="sum",
            fill_value=0,
    ).reset_index()

    report_2.columns.name=None

    # same shape as report_2, but counting transactions instead of summing them
    report_3 = df_b.pivot_table(
            index=["entity_id", "entity_name", "sector"],
            columns="reference",
            values="signed_amount",
            aggfunc="count",
            fill_value=0,
    ).reset_index()

    report_3.columns.name=None

    return report_1, report_2, report_3


# -- 3 BENCHMARK -- 

FINANCING_REFS = ["SWEEP", "TERM-DEPOSIT", "MM-PLACEMENT", "CALL-ACCT", "LOAN", "LOAN-REPAY", "FACILITY"]
TAX_REFS = ["CIT", "VAT201", "PROV-TAX"]

SUPPLIER_REFS = ["PO"]

EMPLOYEE_REFS = ["PAYROLL", "PAYE"]

OUTFLOW_LINES = ["Taxation paid", "Cost of sales and supplier payments", "Employee costs"]


def line_comparison(consolidated_df):
    keys = ["entity_id", "entity_name", "fiscal_year"]
    df = consolidated_df 

    # REVENUE 
    revenue = (df[
        (df["reference"]== "INV") & (df["signed_amount"]>0)
            ]
               .groupby(keys, as_index=False)["signed_amount"].sum()
               .assign(line_item="Total revenue")
               )

    # TAXES 
    taxes = (df[ 
        df["reference"].isin(TAX_REFS)]
               .groupby(keys, as_index=False)["signed_amount"].sum()
               .assign(line_item="Taxation paid")
               )
    taxes["signed_amount"] = taxes["signed_amount"].abs()

    # OPERATING EXP 
    suppliers = (df[ 
        df["reference"].isin(SUPPLIER_REFS)]
                 .groupby(keys, as_index=False)["signed_amount"].sum()
                 .assign(line_item="Cost of sales and supplier payments")
                 )
    suppliers["signed_amount"] = suppliers["signed_amount"].abs()

    operating = (df[~df["reference"].isin(FINANCING_REFS)]
                 .groupby(keys, as_index=False)["signed_amount"].sum()
                 .assign(line_item="Net cash from operating activities")
                 )
    operating["signed_amount"] = operating["signed_amount"]

    # EMPLOYEE_REFS
    employees = (df[ 
        df["reference"].isin(EMPLOYEE_REFS)]
               .groupby(keys, as_index=False)["signed_amount"].sum()
               .assign(line_item="Employee costs")
               )
    employees["signed_amount"] = employees["signed_amount"].abs()

    # TOTAL OUTFLOW
    out = pd.concat([revenue, taxes, suppliers, employees, operating], ignore_index=True)
    return out.rename(columns={"signed_amount":"summation_value"})


def compare_to_results(consolidated_df, path=BENCHMARK_):
    # BRING COMPANY DATA IN 
    worklist = pd.read_csv(path)
    # CREATE REPORTED VALUES COLUMN
    worklist["reported_value"] = pd.to_numeric(worklist["current_value"], errors="coerce")
    # CLEANING THE COLUMN
    worklist = worklist.dropna(subset=["reported_value"])
    
    # ALL OUTFLOWS (MADE POSITIVE VALUE/MAGNITUDE)
    outgoing = worklist["line_item"].isin(OUTFLOW_LINES)
    worklist.loc[outgoing, "reported_value"] = worklist.loc[outgoing, "reported_value"].abs()
    
    #TOTAL LINE 
    calculated = line_comparison(consolidated_df).drop(columns=["entity_name"])


    # MERGE REAL WITH CONSOLIDATED VALUES
    merged = worklist.merge(calculated, on=["entity_id", "fiscal_year", "line_item"], how="left")
    merged["difference"] = merged["summation_value"] - merged["reported_value"]
    merged["pct_of_reported"] = merged["summation_value"]/merged["reported_value"] * 100 

    return merged[["entity_id", "entity_name", "fiscal_year", "line_item", "status", "reported_value", "summation_value", "difference", "pct_of_reported"]]


    




#
# def benchmark_comparison(cons_df, path=BENCHMARK_):
#     outgoing = worklist["line_item"].isin(OUTFLOW_LINES)
#     worklist.loc[outgoing, "reported_value"] = worklist.loc[outgoing, "reported_value"].abs()
#
#     computed = computed_lines(cons_df).drop(columns=["entity_name"])
#
#     merged = worklist.merge(computed, on=["entity_id", "fiscal_year", "line_item"], how="left")
#     merged["difference"] = merged["computed_value"] - merged["reported_value"]
#     merged["pct_of_reported"] = merged["computed_value"] / merged["reported_value"] * 100
#
#     return merged[["entity_id", "entity_name", "fiscal_year", "line_item", "status",
#                    "reported_value", "computed_value", "difference", "pct_of_reported"]]
#





if __name__ == "__main__":
    # == 0 LOADING DATA ==
    transaction_df = load_data(TRANSACTIONS_)
    cross_border_df = load_data(CROSS_BORDER_)
    trade_finance_df = load_data(TRADE_FINANCE_)

    # print(transaction_df.head())
    # print(cross_border_df.head())
    # print(trade_finance_df.head())

    # == 1 VIEWING DATA ==
    reference_type_check(transaction_df)
    reference_type_check(cross_border_df)
    reference_type_check(trade_finance_df)


    # == 2 ENTITY VALUES == 
    consolidated_df = combined_df(transaction_df, cross_border_df, trade_finance_df)
    print(consolidated_df.head())

    summary_profits, summary_references, summary_reference_counts = entity_report(consolidated_df)
    print(summary_profits.head())
    print(summary_references.head())
    print(summary_reference_counts.head())

    # == 3 BENCHMARK COMPARISON ==
    comparison_df = compare_to_results(consolidated_df)
    print(comparison_df.head(20).to_string())
    print("matched:", comparison_df["summation_value"].notna().sum(), "of", len(comparison_df))

    





import io
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Boolean
from sqlalchemy.orm import sessionmaker
from database import Scrubbed, Demographic_Append, Contact_Append, session
from dotenv import dotenv_values
from sqlalchemy.types import Integer, Float, String, Date, Boolean
from sqlalchemy.sql import text
import numpy as np
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import text
from datetime import datetime
import tempfile
import os

username = st.secrets["DATABASE_USERNAME"]
password = st.secrets["DATABASE_PASSWORD"]
dbname = st.secrets["DATABASE_NAME"]
port = st.secrets["DATABASE_PORT"]
host = st.secrets["DATABASE_HOST"]

DATABASE_URL = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"

engine = create_engine(
    DATABASE_URL,
    pool_size=20,       
    max_overflow=10,    
    pool_timeout=30,    
    pool_recycle=1800,  
)
Session = sessionmaker(bind=engine)
db_session = Session()

scrubbed_headers = [
    "OWNER_1_FIRST",
    "OWNER_1_LAST",
    "APN",
    "ACREAGE",
    "GOOD",
    "SUBDIVIDE",
    "Potential Commercial /Commercial",
    "Wetlands Issue",
    "Access issue",
    "Bad Shape",
    "Neighbors own",
    "Slope",
    "House",
    "Transmission Line",
    "No Info Found",
    "Misc. Write note",
    "LATITUDE",
    "LONGITUDE",
    "COUNTY",
    "MAIL_ZIP",
    "MAIL_ADDR",
    "MAIL_CITY",
    "MAIL_STATE",
    "PROP_ZIP",
    "PROP_ADDRESS",
    "PROP_CITY",
    "PROP_STATE",
    "LAND_SQFT",
    "VAL_TRANSFER",
    "UNIT",
    "DATE_TRANSFER",
    "ZONING",
    "OWNER_NAME_1",
    "OWNER_NAME_2",
    "VAL_ASSD_LAND",
    "VAL_MRKT_LAND",
    "CAL_ACREAGE",
    "USE_CODE_MUNI",
    "FLOOD_FACTOR",
    "IMPRV_PCT",
    "LANDUSE_CATEGORY",
    "LANDUSE_CODE",
    "LANDUSE_DESC",
    "LEGAL_1",
    "LOT_NUMBER",
    "OWNERSHIP_STATUS_DESC",
    "TAX_AMOUNT",
    "TAX_YEAR",
    "_SIMPLIFIED",
    "PPA",
    "OFFER_PRICE",
    "County_State",
]

demographic_headers = [
    "DOB",
    "Age",
    "Age Range",
    "Gender",
    "Ethnic Group",
    "Religion",
    "Education Level",
    "Occupation",
    "Language",
    "Marital Status",
    "Working Woman in Household",
    "Senior in Household",
    "Single Parent",
    "Presence of Children",
    "Number of Children",
    "Young Adult in Household",
    "Small Office or Home Office",
    "Online Purchasing Indicator",
    "Online Education"
]

contact_headers = [
    "First Name",
    "Last Name",
    "Mobile Phone 1",
    "Mobile Phone 2",
    "Mobile Phone 3",
    "Mobile Phone 4",
    "Mobile Phone 5",
    "Mobile Phone 6",
    "Landline Phone 1",
    "Landline Phone 2",
    "Landline Phone 3",
    "Landline Phone 4",
    "Landline Phone 5",
    "Landline Phone 6",
    "Voip Phone 1",
    "Voip Phone 2",
    "Voip Phone 3",
    "Voip Phone 4",
    "Voip Phone 5",
    "Voip Phone 6"
]

scrubbed_cols = [
    "GOOD",
    "SUBDIVIDE",
    "Potential Commercial /Commercial",
    "Wetlands Issue",
    "Access issue",
    "Bad Shape",
    "Neighbors own",
    "Slope",
    "House",
    "Transmission Line",
    "No Info Found",
    "Misc. Write note",
]

csv_to_db_mapping = {
    "OWNER_1_FIRST": "owner_first_name",
    "OWNER_1_LAST": "owner_last_name",
    "APN": "apn",
    "ACREAGE": "acreage",
    "GOOD": "is_good",
    "SUBDIVIDE": "is_subdividable",
    "Potential Commercial /Commercial": "is_commercial",
    "Wetlands Issue": "wetland_issues",
    "Access issue": "access_issues",
    "Bad Shape": "bad_shape",
    "Neighbors own": "neighbors_own",
    "Slope": "slope",
    "House": "house",
    "Transmission Line": "transmission_line",
    "No Info Found": "no_info_found",
    "Misc. Write note": "misc_note",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "COUNTY": "county",
    "MAIL_ZIP": "mail_zip",
    "MAIL_ADDR": "mail_addr",
    "MAIL_CITY": "mail_city",
    "MAIL_STATE": "mail_state",
    "PROP_ZIP": "prop_zip",
    "PROP_ADDRESS": "prop_addr",
    "PROP_CITY": "prop_city",
    "PROP_STATE": "prop_state",
    "LAND_SQFT": "land_sqft",
    "VAL_TRANSFER": "val_transfer",
    "UNIT": "unit",
    "DATE_TRANSFER": "date_transfer",
    "ZONING": "zoning",
    "OWNER_NAME_1": "owner_name_1",
    "OWNER_NAME_2": "owner_name_2",
    "VAL_ASSD_LAND": "val_assd_land",
    "VAL_MRKT_LAND": "val_market_land",
    "CAL_ACREAGE": "cal_acreage",
    "USE_CODE_MUNI": "use_code_muni",
    "FLOOD_FACTOR": "flood_factor",
    "IMPRV_PCT": "imprv_pct",
    "LANDUSE_CATEGORY": "landuse_cat",
    "LANDUSE_CODE": "landuse_code",
    "LANDUSE_DESC": "landuse_desc",
    "LEGAL_1": "legal_1",
    "LOT_NUMBER": "lot_number",
    "OWNERSHIP_STATUS_DESC": "ownership_status_desc",
    "TAX_AMOUNT": "tax_amount",
    "TAX_YEAR": "tax_year",
    "PPA": "ppa",
    "OFFER_PRICE": "offer_price",
    "EST MARKET VALUE": "est_market_value",
    "County_State": "county_state"
}

demographic_csv_to_db_mapping = {
    "DOB": "date_of_birth",
    "Age": "age",
    "Age Range": "age_range",
    "Gender": "gender",
    "Ethnic Group": "ethnic_group",
    "Religion": "religion",
    "Education Level": "education_level",
    "Occupation": "occupation",
    "Language": "language",
    "Marital Status": "marital_status",
    "Working Woman in Household": "working_women_household",
    "Senior in Household": "senior_woman_household",
    "Single Parent": "single_parent",
    "Presence of Children": "has_childfren",
    "Number of Children": "no_of_children",
    "Young Adult in Household": "young_adult_household",
    "Small Office or Home Office": "home_office",
    "Online Purchasing Indicator": "online_purchasing",
    "Online Education": "online_education"
}

contact_csv_to_db_mapping = {
    "First Name": "first_name",
    "Last Name": "last_name",
    "Mobile Phone 1": "mob_phone_1",
    "Mobile Phone 2": "mob_phone_2",
    "Mobile Phone 3": "mob_phone_3",
    "Mobile Phone 4": "mob_phone_4",
    "Mobile Phone 5": "mob_phone_5",
    "Mobile Phone 6": "mob_phone_6",
    "Landline Phone 1": "land_phone_1",
    "Landline Phone 2": "land_phone_2",
    "Landline Phone 3": "land_phone_3",
    "Landline Phone 4": "land_phone_4",
    "Landline Phone 5": "land_phone_5",
    "Landline Phone 6": "land_phone_6",
    "Voip Phone 1": "voip_phone_1",
    "Voip Phone 2": "voip_phone_2",
    "Voip Phone 3": "voip_phone_3",
    "Voip Phone 4": "voip_phone_4",
    "Voip Phone 5": "voip_phone_5",
    "Voip Phone 6": "voip_phone_6"
}

SCRUBBED_DESIRED_ORDER = [
    "owner_first_name",
    "owner_last_name",
    "apn",
    "acreage",
    "is_good",
    "is_subdividable",
    "is_commercial",
    "wetland_issues",
    "access_issues",
    "bad_shape",
    "neighbors_own",
    "slope",
    "house",
    "transmission_line",
    "no_info_found",
    "misc_note",
    "latitude",
    "longitude",
    "county",
    "mail_zip",
    "mail_addr",
    "mail_city",
    "mail_state",
    "prop_zip",
    "prop_addr",
    "prop_city",
    "prop_state",
    "land_sqft",
    "val_transfer",
    "unit",
    "date_transfer",
    "zoning",
    "owner_name_1",
    "owner_name_2",
    "val_assd_land",
    "val_market_land",
    "cal_acreage",
    "use_code_muni",
    "flood_factor",
    "imprv_pct",
    "landuse_cat",
    "landuse_code",
    "landuse_desc",
    "legal_1",
    "lot_number",
    "ownership_status_desc",
    "tax_amount",
    "tax_year",
    "simplified",
    "ppa",
    "offer_price",
    "est_market_value",
    "county_state",
    "date_uploaded",
    "date_modified",
    "tags",
    "staging_status"
]

DEMOGRAPHIC_DESIRED_ORDER = [
    "date_of_birth",
    "age",
    "age_range",
    "gender",
    "ethnic_group",
    "religion",
    "education_level",
    "occupation",
    "language",
    "marital_status",
    "working_women_household",
    "senior_woman_household",
    "single_parent",
    "has_childfren",
    "no_of_children",
    "young_adult_household",
    "home_office",
    "online_purchasing",
    "online_education",
    "scrubbed_id",
]

CONTACT_DESIRED_ORDER = [
    "first_name",
    "last_name",
    "mob_phone_1",
    "mob_phone_2",
    "mob_phone_3",
    "mob_phone_4",
    "mob_phone_5",
    "mob_phone_6",
    "land_phone_1",
    "land_phone_2",
    "land_phone_3",
    "land_phone_4",
    "land_phone_5",
    "land_phone_6",
    "voip_phone_1",
    "voip_phone_2",
    "voip_phone_3",
    "voip_phone_4",
    "voip_phone_5",
    "voip_phone_6",
    "scrubbed_id",
]

def equal_enough(new_val, old_val):
    # if both are None or NaN => treat as same
    if (pd.isna(new_val) and pd.isna(old_val)) or (new_val is None and old_val is None):
        return True
    # if numeric, compare after rounding
    if isinstance(new_val, (float, int)) and isinstance(old_val, (float, int)):
        return round(float(new_val), 2) == round(float(old_val), 2)
    # if string => strip/compare
    if isinstance(new_val, str) and isinstance(old_val, str):
        return new_val.strip() == old_val.strip()
    return new_val == old_val



def clean_dataframe_for_sql(df):
    """
    Clean a Pandas DataFrame to ensure compatibility with SQL operations.
    - Replace Pandas `<NA>` with `None`.
    - Replace `nan` in numeric fields with `None`.
    - Convert all `datetime64` fields to `date`.
    """
    df = df.replace({pd.NA: None, np.nan: None})
    
    for column in df.select_dtypes(include=["datetime64"]).columns:
        df[column] = df[column].dt.date  
    
    return df


def escape_single_quotes(value):
    if isinstance(value, str):
        return value.replace("'", "''") 
    return value


def get_combined_reverse_mapping():
    combined_mapping = {
        **csv_to_db_mapping,
        **demographic_csv_to_db_mapping,
        **contact_csv_to_db_mapping
    }
    return {db_col: csv_col for csv_col, db_col in combined_mapping.items()}


def rename_columns_to_original(df):
    """
    Rename DataFrame columns back to their original CSV headers based on the headers array and mapping.
    """
    reverse_mapping = get_combined_reverse_mapping()
    return df.rename(columns=lambda col: reverse_mapping.get(col, col))

def align_csv_to_db(df, mapping):
    """
    Align CSV headers to database column names using a mapping dictionary.
    """
    df = df.rename(columns=mapping)
    return df


def preprocess_all_columns(df, table_model):
    """
    Adjust DataFrame columns to match database schema types.
    Replace NaN and other missing values with None for every column type.
    """
    for column in table_model.__table__.columns:
        col_name = column.name
        if col_name in df.columns:
            if isinstance(column.type, Boolean):
                df[col_name] = df[col_name].map({True: True, False: False, 1: True, 0: False}).where(df[col_name].notna(), None)
      
            elif isinstance(column.type, Integer):
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').where(df[col_name].notna(), None).astype("Int64")
     
            elif isinstance(column.type, Float):
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').where(df[col_name].notna(), None)
            
            elif isinstance(column.type, String):
                df[col_name] = df[col_name].astype(str).where(df[col_name].notna(), None)
           
            elif isinstance(column.type, Date):
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce').dt.date.where(df[col_name].notna(), None)
      
            else:
                df[col_name] = df[col_name].where(df[col_name].notna(), None)
        else:
            df[col_name] = None


    df = df.where(df.notna(), None)

    return df


def map_columns(df, db_columns, type, mapping_dict):
    """
    Map file columns to database schema using the mapping dictionary.
    Automatically map columns if headers match (case-insensitive). Leave unmatched columns for manual mapping.
    Integrates linking fields for demographic and contact types.

    Args:
        df (pd.DataFrame): Uploaded file as a DataFrame.
        db_columns (list): List of database column names.
        type (str): Type of data being mapped (e.g., 'scrubbed', 'demographic', 'contact').
        mapping_dict (dict): Dictionary mapping file headers to database columns.

    Returns:
        dict: Final mapping of database columns to file columns.
    """
    st.write("Map your file columns to the database schema:")
    
    df.columns = df.columns.astype(str)

    linking_fields = [
        "OWNER_1_FIRST", "OWNER_1_LAST", "MAIL_ADDR", "MAIL_ZIP", 
        "MAIL_CITY", "MAIL_STATE", "APN"
    ]
    
    excluded_columns = ["id", "staging_status", "tags", "date_uploaded", "date_modified"]
    
    if type in ["demographic", "contact"]:
        all_columns = linking_fields + [col for col in db_columns if col not in excluded_columns]
    else:
        all_columns = [col for col in db_columns if col not in excluded_columns]

    normalized_file_columns = {str(col).lower().strip(): str(col) for col in df.columns}
    normalized_mapping_dict = {str(key).lower(): value for key, value in mapping_dict.items()}

    initial_mapping = {
        db_col: normalized_file_columns.get(file_col.lower().strip(), None)
        for file_col, db_col in normalized_mapping_dict.items()
    }
    
    for field in linking_fields:
        if field.lower() in normalized_file_columns:
            initial_mapping[field] = normalized_file_columns.get(field.lower())
    
    mapping_df = pd.DataFrame({
        "Database Column": all_columns,
        "File Column Mapping": [
            initial_mapping.get(db_col, None) if db_col in initial_mapping else None
            for db_col in all_columns
        ],
    })


    mapping_df = st.data_editor(
        mapping_df,
        column_config={
            "Database Column": st.column_config.TextColumn("Database Column"),
            "File Column Mapping": st.column_config.SelectboxColumn(
                "File Column Mapping",
                options=["None"] + list(df.columns),  
            ),
        },
        hide_index=True,
        use_container_width=True,
        key=f"data_editor_{type}",
    )
    
    final_mapping = {
        row["Database Column"]: row["File Column Mapping"]
        for _, row in mapping_df.iterrows()
        if row["File Column Mapping"] != "None"
    }

    return final_mapping

def map_scrubbed_id(df):
    """
    Map 'scrubbed_id' for rows in the DataFrame by linking to the Scrubbed table via unique columns.
    Uses dictionary-based lookup for better performance.
    """
    try:
        query = """
        SELECT id, owner_first_name, owner_last_name, mail_zip, mail_addr, mail_city, mail_state, apn
        FROM scrubbed
        """
        with engine.connect() as connection:
            scrubbed_data = pd.read_sql(query, connection)
            
        scrubbed_data = scrubbed_data.fillna("")

        scrubbed_lookup = {
            (row["owner_first_name"], row["owner_last_name"], row["mail_addr"], row["mail_zip"],
             row["mail_city"], row["mail_state"], row["apn"]): row["id"]
            for _, row in scrubbed_data.iterrows()
        }
        df["MAIL_ZIP"] = df["MAIL_ZIP"].astype(str).where(df["MAIL_ZIP"].notna(), None)

        df = df.fillna("")
        df["scrubbed_id"] = df.apply(
            lambda row: scrubbed_lookup.get(
                (row["OWNER_1_FIRST"], row["OWNER_1_LAST"], row["MAIL_ADDR"], row["MAIL_ZIP"],
                 row["MAIL_CITY"], row["MAIL_STATE"], row["APN"])
            ), axis=1
        )

        unmatched = df[df["scrubbed_id"].isnull()]
        if not unmatched.empty:
            st.warning(f"{len(unmatched)} rows could not be matched with the Scrubbed table.")
            st.write("Unmatched Rows Preview:")
            st.dataframe(unmatched.head(20))
          

        return df

    except Exception as e:
        st.error(f"Error while mapping Scrubbed IDs: {e}")
        return df


    
def returning_missing(main_df):
    """
    For each row in the uploaded file:
    1. Check the Scrubbed table for matching data.
    2. Check Demographic table and Contact Append table for matching rows.
    3. Categorize rows into:
       - All data found.
       - Demographic and scrubbed data only.
       - Scrubbed data only.
       - No matching rows.
    4. Allow the user to download each category as a CSV file.
    """
    try:
        main_df = align_csv_to_db(main_df, csv_to_db_mapping)

        main_df = preprocess_all_columns(main_df, Scrubbed)
        scrubbed_query = """
        SELECT id, owner_first_name, owner_last_name, mail_zip, mail_addr, mail_city, mail_state, apn
        FROM scrubbed
        """
        demographic_query = """
        SELECT scrubbed_id
        FROM demographic_append
        """
        contact_query = """
        SELECT scrubbed_id
        FROM contact_append
        """

        with engine.connect() as connection:
            scrubbed_data = pd.read_sql(scrubbed_query, connection)
            st.write("Scrubbed Data Preview:")
            st.dataframe(scrubbed_data.head())
            demographic_data = pd.read_sql(demographic_query, connection)
            contact_data = pd.read_sql(contact_query, connection)

        
        if "id" in main_df.columns:
            main_df = main_df.drop(columns=["id"])
            
        scrubbed_merged = pd.merge(
            main_df,
            scrubbed_data,
            how="inner",
            left_on=["owner_first_name", "owner_last_name", "mail_addr", "mail_zip", "mail_city", "mail_state", "apn"],
            right_on=["owner_first_name", "owner_last_name", "mail_addr", "mail_zip", "mail_city", "mail_state", "apn"]
        )
        
        scrubbed_merged = scrubbed_merged[[col.name for col in Scrubbed.__table__.columns]]
        
        st.write("Scrubbed Merged DataFrame Preview:")
        st.dataframe(scrubbed_merged.head())
        st.write("Columns in Merged DataFrame:")
        st.write(scrubbed_merged.columns.tolist())

        scrubbed_merged["scrubbed_found"] = scrubbed_merged["id"].notna()

        scrubbed_merged["scrubbed_id"] = scrubbed_merged["id"].astype("Int64")
        demographic_found = scrubbed_merged["scrubbed_id"].isin(demographic_data["scrubbed_id"])
        contact_found = scrubbed_merged["scrubbed_id"].isin(contact_data["scrubbed_id"])

        # Categorize rows
        all_data_found = scrubbed_merged[demographic_found & contact_found]
        remaining_rows = scrubbed_merged[~scrubbed_merged.index.isin(all_data_found.index)]

        demographic_and_scrubbed = remaining_rows[remaining_rows["scrubbed_found"] & demographic_found]
        remaining_rows = remaining_rows[~remaining_rows.index.isin(demographic_and_scrubbed.index)]

        scrubbed_only = remaining_rows[remaining_rows["scrubbed_found"]]
        remaining_rows = remaining_rows[~remaining_rows.index.isin(scrubbed_only.index)]

        no_match = remaining_rows

        all_data_found = rename_columns_to_original(all_data_found)
        demographic_and_scrubbed = rename_columns_to_original(demographic_and_scrubbed)
        scrubbed_only = rename_columns_to_original(scrubbed_only)
        no_match = rename_columns_to_original(no_match)

        # Display results and provide download links
        st.subheader("All Data Found")
        st.dataframe(all_data_found)
        st.download_button(
            label="Download All Data Found as CSV",
            data=all_data_found.to_csv(index=False).encode('utf-8'),
            file_name="all_data_found.csv",
            mime="text/csv"
        )

        st.subheader("Demographic and Scrubbed Data Only")
        st.dataframe(demographic_and_scrubbed)
        st.download_button(
            label="Download Demographic and Scrubbed Data as CSV",
            data=demographic_and_scrubbed.to_csv(index=False).encode('utf-8'),
            file_name="demographic_and_scrubbed.csv",
            mime="text/csv"
        )

        st.subheader("Scrubbed Data Only")
        st.dataframe(scrubbed_only)
        st.download_button(
            label="Download Scrubbed Data Only as CSV",
            data=scrubbed_only.to_csv(index=False).encode('utf-8'),
            file_name="scrubbed_only.csv",
            mime="text/csv"
        )

        st.subheader("No Matching Rows")
        st.dataframe(no_match)
        st.download_button(
            label="Download No Matching Rows as CSV",
            data=no_match.to_csv(index=False).encode('utf-8'),
            file_name="no_matching_rows.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred while checking for existing data: {e}")
        
def determine_staging_status(uploaded_df, table_name, table_model):
    """
    Determine if records in the uploaded DataFrame are "New", "Updated", or "Unchanged".
    Uses set-based lookups for faster comparison instead of slow DataFrame merging.
    Adjusts lookup strategy based on table type.
    """
    if table_name == "scrubbed":
        primary_keys = [
            "owner_first_name", "owner_last_name", "mail_zip",
            "mail_addr", "mail_city", "mail_state",
             "apn"
        ]
    elif table_name in ["demographic_append", "contact_append"]:
        primary_keys = ["scrubbed_id"]  
    else:
        st.error(f"Unknown table name: {table_name}")
        return uploaded_df
    
    duplicates_in_upload = (
        uploaded_df.groupby(primary_keys)
                .size()
                .reset_index(name='count')
                .query('count > 1')
    )
    print("Duplicates in uploaded_df on PK:", duplicates_in_upload)


    # Fetch only necessary columns from the database
    query = f"SELECT * FROM {table_name}"
    
    with engine.connect() as connection:
        existing_df = pd.read_sql(query, connection)
        print("Existing DF Length: ",len(existing_df))
        existing_df = existing_df.drop_duplicates(subset=primary_keys)
        print("Existing DF Length after drop dupes: ",len(existing_df))

    if existing_df.empty:
        uploaded_df["staging_status"] = "New"
        return uploaded_df
    
    if "mail_zip" in primary_keys:
        existing_df["mail_zip"] = existing_df["mail_zip"].astype(str)
        uploaded_df["mail_zip"] = uploaded_df["mail_zip"].astype(str)

    print("existing_df.shape =", existing_df.shape)

    existing_dict = {}
    existing_records = existing_df.to_dict(orient="records")
    for record in existing_records:
        key_tuple = tuple(record[pk] for pk in primary_keys)
        existing_dict[key_tuple] = record  

    staging_status = []
    all_columns = uploaded_df.columns.tolist()
    non_pk_cols = [c for c in all_columns if c not in primary_keys and c not in ["latitude", "longitude"]]
    uploaded_df["misc_note"] = uploaded_df["misc_note"].where(uploaded_df["misc_note"].notna(), None)
    uploaded_df["prop_zip"] = uploaded_df["prop_zip"].astype(str).str.strip()
    uploaded_df["date_transfer"] = pd.to_datetime(uploaded_df["date_transfer"], errors="coerce").dt.date

    for _, row in uploaded_df.iterrows():
        pk_tuple = tuple(row[pk] for pk in primary_keys)

        if pk_tuple not in existing_dict:
            staging_status.append("New")
        else:
            db_row = existing_dict[pk_tuple]
            is_updated = False
            for col in non_pk_cols:
                # compare new vs old
                new_val = row[col]
                old_val = db_row.get(col) 
                if not equal_enough(new_val, old_val):
                    print(f"Diff in col '{col}': new={new_val}, old={old_val}")
                    is_updated = True
                    break
            
            if is_updated:
                staging_status.append("Updated")
            else:
                staging_status.append("Unchanged")

    uploaded_df["staging_status"] = staging_status
    st.dataframe(uploaded_df.head(4))
    return uploaded_df

def upload_scrubbed_data_to_db(df, table_model, table_name, column_mapping):
    """
    Uploads DataFrame to 'scrubbed' using staging_status logic:
      - "New" => Insert only.
      - "Updated" => Update only.
    Also handles date_uploaded for Updated vs. New.
    """
    try:
        if df.empty:
            st.warning(f"No data to upload to '{table_name}'.")
            return

        # 1. Define primary keys for scrubbed
        primary_keys = [
            "owner_first_name", "owner_last_name", "mail_zip",
            "mail_addr", "mail_city", "mail_state", "apn"
        ]
        # If 'tags' not present, ensure it's in the DataFrame
        if "tags" not in df.columns:
            df["tags"] = None

        # 2. Remap columns to DB schema
        selected_columns = {
            file_col: db_col
            for db_col, file_col in column_mapping.items()
            if file_col in df.columns
        }
        selected_cols = list(selected_columns.keys())
        
        # Also ensure 'tags' is included if it's not mapped
        if "tags" in df.columns and "tags" not in selected_cols:
            selected_cols.append("tags")

        df = df[selected_cols].rename(columns=selected_columns)

        # 3. Determine staging status (New / Updated / Unchanged)
        print("Row count before staging:", len(df))
        df = determine_staging_status(df, table_name, table_model)
        print("Row count after staging:", len(df))

        # 4. Pull existing date_uploaded from DB
        with engine.connect() as conn:
            existing_dates = pd.read_sql("""
                SELECT owner_first_name, owner_last_name, mail_zip,
                       mail_addr, mail_city, mail_state, apn,
                       date_uploaded
                FROM scrubbed
            """, conn)
        dates_dict = {}
        for rec in existing_dates.to_dict(orient="records"):
            key_tuple = (
                rec["owner_first_name"],
                rec["owner_last_name"],
                rec["mail_zip"],
                rec["mail_addr"],
                rec["mail_city"],
                rec["mail_state"],
                rec["apn"],
            )
            dates_dict[key_tuple] = rec["date_uploaded"]

        # Fill in date_uploaded for Updated/Unchanged
        for idx, row in df.iterrows():
            if row["staging_status"] in ["Updated", "Unchanged"]:
                key_tuple = (
                    row["owner_first_name"],
                    row["owner_last_name"],
                    row["mail_zip"],
                    row["mail_addr"],
                    row["mail_city"],
                    row["mail_state"],
                    row["apn"],
                )
                old_date = dates_dict.get(key_tuple, None)
                df.at[idx, "date_uploaded"] = old_date

        # 5. Set date_uploaded for New
        current_date = datetime.now().date()
        if "date_uploaded" not in df.columns:
            df["date_uploaded"] = None
        if "date_modified" not in df.columns:
            df["date_modified"] = None
        df.loc[df["staging_status"] == "New", ["date_uploaded","date_modified"]] = current_date
        df.loc[df["staging_status"].isin(["Updated", "Unchanged"]), "date_modified"] = current_date

        # 6. Preprocess columns & reorder
        df = preprocess_all_columns(df, table_model)
        df = clean_dataframe_for_sql(df)
        df = df.drop(columns=["id"], errors="ignore")
        df = df[SCRUBBED_DESIRED_ORDER]  # reorder columns for scrubbed

        # 7. If date_transfer exists, fix
        if "date_transfer" in df.columns:
            df["date_transfer"] = pd.to_datetime(df["date_transfer"], errors="coerce").dt.date

        # 8. Create temp staging table & COPY
        temp_table = f"{table_name}_staging"
        with engine.begin() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {temp_table};"))
            connection.execute(text(f"CREATE TEMP TABLE {temp_table} (LIKE {table_name} INCLUDING DEFAULTS);"))
            connection.execute(text(f"ALTER TABLE {temp_table} DROP COLUMN IF EXISTS id;"))

            columns_list = df.columns.tolist()
            print("DataFrame columns (in code order):", columns_list)

            raw_conn = connection.connection
            cursor = raw_conn.cursor()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
                df.to_csv(tmpfile.name, index=False, header=False, sep="\t", na_rep="\\N", quoting=3)
                tmpfile_path = tmpfile.name

            copy_sql = f"""
                COPY {temp_table} ({", ".join(columns_list)})
                FROM STDIN
                WITH (
                    FORMAT csv,
                    DELIMITER E'\t',
                    NULL '\\N',
                    QUOTE '\"',
                    HEADER FALSE
                )
            """
            with open(tmpfile_path, "r") as f2:
                cursor.copy_expert(copy_sql, f2)
            os.remove(tmpfile_path)

            # INSERT only 'New' rows
            pk_conditions = " AND ".join([f"{table_name}.{col} = {temp_table}.{col}" for col in primary_keys])
            insert_sql = f"""
                INSERT INTO {table_name} ({', '.join(columns_list)})
                SELECT {', '.join(columns_list)}
                FROM {temp_table}
                WHERE staging_status = 'New'
                  AND NOT EXISTS (
                      SELECT 1 FROM {table_name}
                      WHERE {pk_conditions}
                  );
            """
            connection.execute(text(insert_sql))

            # UPDATE only 'Updated'
            update_assignments = []
            skip_cols = set(primary_keys + ["staging_status"])
            for col in columns_list:
                if col in skip_cols:
                    continue
                if col == "tags":
                    update_assignments.append(f"tags = COALESCE({table_name}.tags, {temp_table}.tags)")
                else:
                    update_assignments.append(f"{col} = {temp_table}.{col}")

            update_sql = f"""
                UPDATE {table_name}
                SET {', '.join(update_assignments)}
                FROM {temp_table}
                WHERE {temp_table}.staging_status = 'Updated'
                  AND {" AND ".join([f"{table_name}.{pk} = {temp_table}.{pk}" for pk in primary_keys])};
            """
            connection.execute(text(update_sql))

            connection.execute(text(f"DROP TABLE IF EXISTS {temp_table};"))

        st.dataframe(df.head(10))
        st.success(f"Successfully upserted {len(df)} 'scrubbed' rows via \copy and batch UPDATE.")

    except Exception as e:
        st.error(f"Error in upload_scrubbed_data_to_db(): {e}")
        
def upsert_demo_or_contact_data(
    df: pd.DataFrame,
    table_model,               
    table_name: str,          
    column_mapping: dict       
):
    try:
        # 1) Check if DF is empty
        if df.empty:
            st.warning(f"No data to upload to '{table_name}'.")
            return

        primary_key = "scrubbed_id"

        # 3) Rename CSV columns -> DB columns
        #    (We invert the column_mapping to find which columns you want to keep)
        selected_columns = {
            file_col: db_col
            for db_col, file_col in column_mapping.items()
            if file_col in df.columns
        }
        selected_cols = list(selected_columns.keys())  
        df = df[selected_cols].rename(columns=selected_columns)

        # 4) Map scrubbed_id from 'scrubbed' so each row references an existing row
        df = map_scrubbed_id(df)
        df = df[df["scrubbed_id"].notnull()].copy()
        if df.empty:
            st.warning(f"No matching rows found for {table_name}: all scrubbed_id are null.")
            return

        # 5) Preprocess columns
        df = preprocess_all_columns(df, table_model)
        df = clean_dataframe_for_sql(df)
        df = df.drop(columns=["id"], errors="ignore")  # remove 'id' if present

        # Reorder columns to match final DB structure
        if table_name == "demographic_append":
            df = df[DEMOGRAPHIC_DESIRED_ORDER]
        elif table_name == "contact_append":
            df = df[CONTACT_DESIRED_ORDER]

        if df.empty:
            st.warning(f"No valid columns or data to insert into {table_name}.")
            return

        # 6) Create a temp staging table
        temp_table = f"{table_name}_staging"
        with engine.begin() as connection:
            # drop old staging
            connection.execute(text(f"DROP TABLE IF EXISTS {temp_table};"))
            # create new staging from final table
            connection.execute(
                text(f"CREATE TEMP TABLE {temp_table} (LIKE {table_name} INCLUDING DEFAULTS);")
            )
            # remove 'id' col if it exists
            connection.execute(text(f"ALTER TABLE {temp_table} DROP COLUMN IF EXISTS id;"))

            columns_list = df.columns.tolist()

            # 7) COPY
            raw_conn = connection.connection
            cursor = raw_conn.cursor()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
                df.to_csv(tmpfile.name, index=False, header=False, sep="\t",
                          na_rep="\\N", quoting=3)
                tmpfile_path = tmpfile.name

            copy_sql = f"""
                COPY {temp_table} ({", ".join(columns_list)})
                FROM STDIN
                WITH (
                    FORMAT csv,
                    DELIMITER E'\t',
                    NULL '\\N',
                    QUOTE '\"',
                    HEADER FALSE
                )
            """
            with open(tmpfile_path, "r") as f2:
                cursor.copy_expert(copy_sql, f2)
            os.remove(tmpfile_path)

            # 8) Build the DO UPDATE assignment set, skipping the PK
            skip_cols = [primary_key]
            update_assignments = []
            for col in columns_list:
                if col not in skip_cols:
                    update_assignments.append(f"{col} = EXCLUDED.{col}")
            update_str = ", ".join(update_assignments)

            # 9) Insert with ON CONFLICT
            insert_sql = f"""
                INSERT INTO {table_name} ({', '.join(columns_list)})
                SELECT {', '.join(columns_list)}
                FROM {temp_table}
                ON CONFLICT ({primary_key})
                DO UPDATE
                SET {update_str};
            """
            connection.execute(text(insert_sql))

            connection.execute(text(f"DROP TABLE IF EXISTS {temp_table};"))

        st.dataframe(df.head(10))
        st.success(f"Upserted {len(df)} rows into '{table_name}' (ON CONFLICT({primary_key}) DO UPDATE).")

    except Exception as e:
        st.error(f"Error in upsert_demo_or_contact_data(): {e}")

def main():
    st.title("LandBH Database Uploader")

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel or CSV file:", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            # Read the file
            file_ext = uploaded_file.name.split(".")[-1]
            if file_ext == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_ext == "xlsx":
                df = pd.read_excel(uploaded_file)
           
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())

            include_scrubbed = st.checkbox("Includes Scrubbed Data?")
            include_demographic = st.checkbox("Includes Demographic Append Data?")
            include_contact = st.checkbox("Includes Contact Append Data?")
            match_with_existing_data = st.checkbox("Match With Existing Data In Database?")

            if include_scrubbed:
                st.write("Add Tags for the Uploaded Data:")
                tags_input = st.text_input("Enter tags (comma-separated):", "")
                tags_list = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

                if tags_list:
                    df["tags"] = ", ".join(tags_list)
                else:
                    df["tags"] = None
                st.subheader("Scrubbed Data Mapping")
                db_columns = [col.name for col in Scrubbed.__table__.columns if col.name not in ["id", "date_uploaded", "date_modified", "tags"]]
                
                scrubbed_mapping = map_columns(df, db_columns, "scrubbed", csv_to_db_mapping)
                

                if st.button("Upload Scrubbed Data"):
                    upload_scrubbed_data_to_db(df, Scrubbed, "scrubbed", scrubbed_mapping)

            if include_demographic:
                st.subheader("Demographic Append Data Mapping")
                db_columns = [col.name for col in Demographic_Append.__table__.columns if col.name not in ["id", "scrubbed_id"]]
                demographic_mapping = map_columns(df, db_columns, "demographic", demographic_csv_to_db_mapping)

                if st.button("Upload Demographic Append Data"):
                    upsert_demo_or_contact_data(df, Demographic_Append, "demographic_append", demographic_mapping)

            if include_contact:
                st.subheader("Contact Append Data Mapping")
                db_columns = [col.name for col in Contact_Append.__table__.columns if col.name not in ["id", "scrubbed_id"]]
                contact_mapping = map_columns(df, db_columns, "contact", contact_csv_to_db_mapping)

                if st.button("Upload Contact Append Data"):
                    upsert_demo_or_contact_data(df, Contact_Append, "contact_append", contact_mapping)

            if match_with_existing_data:
                st.subheader("Match With DB Records")
                db_columns = [col.name for col in Scrubbed.__table__.columns if col.name not in ["id"]]
                scrubbed_mapping = map_columns(df, db_columns, "matching")

                if st.button("Match With Existing Records"):
                    returning_missing(df)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()


# ***Step 0***: place all csv files, organised by day, in training_set folder with the following structure:
## Create subfolders for each day:
### `training_set/20180715/, training_set/20180716/, training_set/20180717/, ...`
## In each of these folders, place all csv files from the MSSD dataset for that day. In `training_set/20180715`, we should have:
### `log_0_20180715_000000000000.csv`, `log_1_20180715_000000000000.csv`, `log_2_20180715_000000000000.csv`, ..., `log_9_20180715_000000000000.csv`

# ***Step 1***: convert all csv files to parquet
# ***Step 2***: convert each day to a single SqLite database

import os
import glob
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

# downscale fields of Pandas dataframes to reduce memory
def __downscale_field(df, field_name):
    int_cols = ["session_position", "session_length", "context_switch", "no_pause_before_play", "short_pause_before_play", "long_pause_before_play", "hist_user_behavior_n_seekfwd", "hist_user_behavior_n_seekback", "hour_of_day"]
    float_cols = ["listening_pattern"]

    if field_name in int_cols:
        df[field_name] = pd.to_numeric(df[field_name], downcast="integer")
    elif field_name in float_cols:
        df[field_name] = pd.to_numeric(df[field_name], downcast="float")
    
    return df

# remove sessions from dataset where the skip pattern F,T,F,F is found OR None values are present
def __filter_dataframe_unobserved_skip_pattern(df):
    session_ids_to_remove = df.loc[(df["skip_1"] == False) & (df["skip_2"] == True) & (df["skip_3"] == False) & (df["not_skipped"] == False)].session_id.unique()
    df = df[~df.session_id.isin(session_ids_to_remove)]
    
    session_ids_to_remove = df.loc[(df["skip_1"].isnull()) | (df["skip_2"].isnull()) | (df["skip_3"].isnull()) | (df["not_skipped"].isnull())].session_id.unique()
    df = df[~df.session_id.isin(session_ids_to_remove)]

    return df

# convert skip pattern to new row by assigning 1/2/3/4/5 ID (listening_pattern)
def __label_skip_pattern(df):
    df.loc[((df["skip_1"] == True) & (df["skip_2"] == True) & (df["skip_3"] == True) & (df["not_skipped"] == False)), "listening_pattern"] = 1 # "Very Very Briefly"
    df.loc[((df["skip_1"] == False) & (df["skip_2"] == True) & (df["skip_3"] == True) & (df["not_skipped"] == False)), "listening_pattern"] = 2 # "Very Briefly"
    df.loc[((df["skip_1"] == False) & (df["skip_2"] == False) & (df["skip_3"] == True) & (df["not_skipped"] == False)), "listening_pattern"] = 3 # "Briefly"
    df.loc[((df["skip_1"] == False) & (df["skip_2"] == False) & (df["skip_3"] == False) & (df["not_skipped"] == False)), "listening_pattern"] = 4 # "Most"
    df.loc[((df["skip_1"] == False) & (df["skip_2"] == False) & (df["skip_3"] == False) & (df["not_skipped"] == True)), "listening_pattern"] = 5 # "All"

    return df

def convert_csv_to_parquet():
    # retrieve all days for selected week
    days = sorted(glob.glob("data/training_set/*"))
    for day in days:
        # all logs for each day
        logs = sorted(glob.glob(day + "/*"))

        for log in logs:
            # read csv log
            print("Read \"{0}\"".format(log))
            df = pd.read_csv(log)

            # change extension in filepath and save as parquet file
            new_log = os.path.splitext(log)[0] + ".parquet"
            print("Create \"{0}\"".format(new_log))
            df.to_parquet(new_log)
            
            # delete csv file from local disk
            print("Delete \"{0}\"".format(log))
            os.remove(log)

def create_dbs():
    days = sorted(glob.glob("data/training_set/*"))
    for day in days:
        print("Reading Day: {0}".format(day))

        # create db connection
        db_name = "sqlite:///data/{0}.db".format(os.path.basename(day))
        db = create_engine(db_name, echo=False)

        # for every day, get all logs
        logs = sorted(glob.glob(day + "/log_*"))
        for log in logs:
            # load individual log (parquet)
            df = pd.read_parquet(log)
            
            # remove sessions with unboserved skip pattern
            df = __filter_dataframe_unobserved_skip_pattern(df)

            # convert skip flags to ID representation (1-5 scale)
            df = __label_skip_pattern(df)
            
            # remove unwanted features
            df.drop("skip_1", axis=1, inplace=True)
            df.drop("skip_2", axis=1, inplace=True)
            df.drop("skip_3", axis=1, inplace=True)
            df.drop("not_skipped", axis=1, inplace=True)
            df.drop("date", axis=1, inplace=True)
            
            # downscale all columns (int and floats)
            for col in df.columns.values:
                df = __downscale_field(df, col)

            # store to db, remove index column, and specify datatypes
            db_dtypes = {
                "session_id": sqlalchemy.types.NVARCHAR(length=50),
                "session_position": sqlalchemy.types.INT(),
                "session_length": sqlalchemy.types.INT(),
                "track_id_clean": sqlalchemy.types.NVARCHAR(length=50),
                "context_switch": sqlalchemy.types.INT(),
                "no_pause_before_play": sqlalchemy.types.INT(),
                "short_pause_before_play": sqlalchemy.types.INT(),
                "long_pause_before_play": sqlalchemy.types.INT(),
                "hist_user_behavior_n_seekfwd": sqlalchemy.types.INT(),
                "hist_user_behavior_n_seekback": sqlalchemy.types.INT(),
                "hist_user_behavior_is_shuffle": sqlalchemy.types.Boolean(),
                "hour_of_day": sqlalchemy.types.INT(),
                "premium": sqlalchemy.types.Boolean(),
                "context_type": sqlalchemy.types.NVARCHAR(length=50),
                "hist_user_behavior_reason_start": sqlalchemy.types.NVARCHAR(length=50),
                "hist_user_behavior_reason_end": sqlalchemy.types.NVARCHAR(length=50),
                "listening_pattern": sqlalchemy.types.Float(),
            }
            df.to_sql("sessions", db, if_exists="append", index=False, dtype=db_dtypes)

        # close connection to db once done
        db.dispose()

if __name__ == "__main__":
    convert_csv_to_parquet()
    create_dbs()

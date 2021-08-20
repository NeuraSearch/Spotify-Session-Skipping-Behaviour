import os
import glob
import pandas as pd

from sqlalchemy import create_engine

# select time window
# "night": 0-5
# "morning": 6-11
# "afternoon": 12-17
# "evening": 18-23
# "all"
def __hour_range(day_time):
    if day_time == "night":
        return (0, 5)
    elif day_time == "morning":
        return (6, 11)
    elif day_time == "afternoon":
        return (12, 17)
    elif day_time == "evening":
        return (18, 23)
    elif day_time == "all":
        return (0, 23)
    else:
        raise Exception("Unrecognised day_time value. It has to be 'night', 'morning', 'afternoon', 'evening', 'all'")

def __get_all_files():
    return sorted(glob.glob("data/*.db"))

def __get_weekdays_weekends_files():
    # we ignore week10 days because it is not a full week
    week_10_days = ["20180916", "20180917", "20180918"]

    weekdays = []
    weekends = []
    for day_path in __get_all_files():
        day = os.path.splitext(os.path.basename(day_path))[0]

        # ignore Week10 days if ignore_week_10 is True
        if (day in week_10_days):
            continue

        # if weekend, put day in weekend list, otherwise in weekday
        day_of_week = pd.to_datetime(day).weekday()
        if (day_of_week == 5) or (day_of_week == 6):
            weekends.append(day_path)
        else:
            weekdays.append(day_path)
    
    return weekdays, weekends

def __create_select_query(context_types):
    if context_types:
        return "SELECT GROUP_CONCAT(DISTINCT context_type) AS context_types, GROUP_CONCAT(listening_pattern) AS listening_pattern"

    # if none of the above, return standard query
    return "SELECT GROUP_CONCAT(listening_pattern) AS listening_pattern"

# splitting and filtering of context_types (e.g. only select sessions that
# have a listening context type equal to catalog). It allows for selection of multiple context types
def __process_context_types_filtering(dataframe, context_types):
    # convert to list first, then back to string and check length
    dataframe["context_types"] = dataframe["context_types"].str.split(",")

    # select only those sessions with `context_types` equal to CONTEXT_TYPES
    # (with sorting to include permutations). Then drop `context_types` column
    dataframe = dataframe[[sorted(x) == context_types for x in dataframe["context_types"]]]
    dataframe = dataframe.drop("context_types", axis=1)

    return dataframe

def __split_listening_pattern_in_dataframe(dataframe):
    # split column `listening_pattern` by comma
    dataframe = pd.concat([dataframe, dataframe["listening_pattern"].str.split(',', expand=True).astype("float32")], axis=1)
    dataframe.drop("listening_pattern", axis=1, inplace=True)

    # rename newly generated columns as: 0 --> pos1, 1 --> pos2
    new_headers = []
    for i in range(1, len(dataframe.columns) + 1):
        new_headers.append("pos{0}".format(i))
    dataframe.columns = new_headers
    dataframe = dataframe.apply(pd.to_numeric)

    return dataframe

def __gather_daily_dataframe(day, session_length, context_types, day_time="all"):
    # get range of hours
    hour_start, hour_end = __hour_range(day_time)

    # connect to db
    db_name = "sqlite:///{0}".format(day)
    db = create_engine(db_name, echo=False)

    # run sql query and store result to dataframe
    select_statement = __create_select_query(context_types)
    dataframe = pd.read_sql(
        """
        {0}
        FROM(
            SELECT session_id, hour_of_day, context_type, listening_pattern
            FROM sessions
            WHERE session_length == {1}
            ORDER BY session_id, session_position
        )
        GROUP BY session_id
        HAVING ROUND(AVG(hour_of_day), 0) BETWEEN {2} AND {3}
        """.format(select_statement, session_length, hour_start, hour_end),
        con=db
    )

    # close connection
    db.dispose()

    if context_types:
        dataframe = __process_context_types_filtering(dataframe, context_types)

    # process dataframe to correct structure
    dataframe = __split_listening_pattern_in_dataframe(dataframe)

    return dataframe

# this method returns a single dataframe which is concatenation of all dataframes in which experimental conditions are applied.
# For example, it returns all mornings in a single dataframe
def __gather_single_dataframe(days, session_length, context_types, day_time):
    dataframe = None
    for day in days:
        print("Reading: {0}".format(day))

        # get full data for that day
        df = __gather_daily_dataframe(day, session_length, context_types, day_time=day_time)

        # update dataframe with new sample. Set if first time, otherwise append to existing dataframe
        if dataframe is None:
            dataframe = df
        else:
            dataframe = dataframe.append(df, ignore_index=True)

    return dataframe

def generate_dataframe(_type, session_length, context_types, dataframe_path):
    # get dataframe based on experimental conditions
    if _type == "all":
        dataframe = __gather_single_dataframe(__get_all_files(), session_length, context_types, _type)
    if _type == "weekday":
        days = __get_weekdays_weekends_files()[0]
        dataframe = __gather_single_dataframe(days, session_length, context_types, "all")
    elif _type == "weekend":
        days = __get_weekdays_weekends_files()[1]
        dataframe = __gather_single_dataframe(days, session_length, context_types, "all")
    elif (_type == "morning") or (_type == "afternoon") or (_type == "evening") or (_type == "night"):
        dataframe = __gather_single_dataframe(__get_all_files(), session_length, context_types, _type)
    
    # save dataframe to file
    dataframe.to_parquet(dataframe_path)

    return dataframe

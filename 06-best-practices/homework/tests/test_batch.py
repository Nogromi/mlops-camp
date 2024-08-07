from datetime import datetime
import pandas as pd
from pandas.testing import assert_frame_equal
import batch

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    categorical = ['PULocationID', 'DOLocationID']

    df = pd.DataFrame(data, columns=columns)
    actual_df=batch.prepare_data(df, categorical=categorical)
    print(actual_df)

    expected_data = [
        ("-1", "-1", dt(1, 1), dt(1, 10), 9.0),
        ("1", "1", dt(1, 2), dt(1, 10), 8.0),
    ]
    expected_columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "duration",
    ]

    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    assert actual_df.shape[0]==2
    assert actual_df.columns.tolist() == expected_df.columns.tolist()
    assert_frame_equal(actual_df, expected_df)
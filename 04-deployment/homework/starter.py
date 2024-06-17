#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import sys


def read_data(filename, categorical):
        df = pd.read_parquet(filename)
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df


def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3
    print(year, month)
    year=f'{year:04d}'
    month=f'{month:02d}'
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    print('get data')
    path=f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
    print('path',path)
    df = read_data(path,
                   categorical)

    print('transform data')

    dicts = df[categorical].to_dict(orient='records')

    X_val = dv.transform(dicts)
    print('predict data')

    y_pred = model.predict(X_val)
    print("y_pred mean", np.mean(y_pred))

    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
    d={'ride_id':df['ride_id'].values, 'duration':y_pred  }
    df_result=pd.DataFrame(data=d)
    print(df_result.head())


    output_file=f'{month}_{year}_preds.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )




if __name__ == '__main__':
    run()
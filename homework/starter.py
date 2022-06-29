import pickle
import pandas as pd
import sys

year = int(sys.argv[1])
month = int(sys.argv[2])


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def run():
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('mean of predicted duration: ',y_pred.mean())

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_results = pd.DataFrame()

    df_results['result'] = y_pred
    df_results['ride_id'] = df['ride_id']

    df_results.to_parquet(
        f'output_file_{year:04d}_{month:02d}.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )
    print('Done')

if __name__ == '__main__':
    run()

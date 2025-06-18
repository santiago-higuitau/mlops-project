
import pickle
import argparse
import pandas as pd


#-----------------------------------------------------------------------------*
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
args = parser.parse_args()
#-----------------------------------------------------------------------------*
categorical = ['PULocationID', 'DOLocationID']

def read_data(url):
    print('Downloading dataset...')
    df = pd.read_parquet(url)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def make_predictions(url, model, dv):
    df = read_data(url)
    print(f'Dataset for {year}-{month:02d} downloaded successfully!')
    print(f'Dimensions: {df.shape=}')

    print('Generating predictions...')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred
#-----------------------------------------------------------------------------*

year = args.year
month = args.month
dataset = 'yellow'

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# Execution:
url = (
    'https://d37ci6vzurychx.cloudfront.net/trip-data/'
    f'{dataset}_tripdata_{year}-{month:02d}.parquet'
)


predictions = make_predictions(url, model, dv)
print(
    f'Mean predicted duration of {year}'
    f'-{month:02d}: {predictions.mean():.2f}'
)

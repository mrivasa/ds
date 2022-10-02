import pandas as pd
import numpy as np
#from faker import Faker
from sklearn.neighbors import BallTree
from geopy import distance

import functools
import time

# Timing Decorator
def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

# def generate_data(length):
#     '''
#         Genearte data frame with random lat/lon data 
#         Data with same length is always identical since we use same initial seed
#     '''
#     Faker.seed(0)
#     fake = Faker()

#     id = list(range(length))
#     # First two fields of fake.local_latlng are lat & lon as string
#     # Generate vector of fake.local_latlng then unpack out lat/lon array
#     lat, lon = list(zip(*[fake.local_latlng() for _ in range(length)]))[:2]
    
#     # Convert strings to float
#     lat = [float(x) for x in lat]
#     lon = [float(x) for x in lon]
    
#     return pd.DataFrame({'point_id':id, 'lat':lat, 'lon':lon})

def generate_balltree(df):
    '''
        Generate Balltree using customize distance (i.e. Geodesic distance)
    '''
    return  BallTree(df[['Latitude', 'Longitude']].values, metric=lambda u, v: distance.distance(u, v).miles)

@timer
def find_matches(tree, df):
    '''
        Find closest matches in df to items in tree
    '''
    distances, indices = tree.query(df[['lat', 'lon']].values, k = 1)
    df['min_dist'] = distances
    df['min_loc'] = indices
    
    return df

@timer
def find_min_op(df1, df2):
    ' OP solution (to compare timing) '

    for x in range(len(df1)):
        #loc_id=df1.iloc[x]['loc_id'] # not used
        pt1=(df1.iloc[x]['Latitude'],df1.iloc[x]['Longitude'])
        for y in range(len(df2)):
            pt2=(df2.iloc[y]['latitude'],df2.iloc[y]['longitude'])
            dist=distance.distance(pt1,pt2).miles
            df2.loc[y,x]=dist

    # determining minimum distance and label:
    temp_cols=list(range(len(df1)))
    df2['min_dist']=df2[temp_cols].min(axis=1)
    df2['min_loc']=df2[temp_cols].idxmin(axis=1)

    # removing extra columns:
    df2 = df2.drop(columns = temp_cols)
    
    return df2

l1, l2 = 100, 1000
df2 = pd.read_csv("asian joro observations-260012.csv", delimiter=',')
df1 = pd.read_csv("asiaWorldClimAll.csv", delimiter=',')
tree = generate_balltree(df1)
find_matches(tree, df2)

find_min_op(df1, df2)
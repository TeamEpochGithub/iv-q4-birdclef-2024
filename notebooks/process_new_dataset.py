# %%
import pandas as pd
# %%

og_df = pd.read_csv('data/raw/train_data/train_metadata.csv.zip')
# download from https://www.kaggle.com/datasets/ludovick/birdclef2024-additional-mp3
add_df = pd.read_csv('data/raw/train_data/BirdClef2024_additional.csv.zip')

# %%
more_data = add_df[add_df['primary_label'].isin(og_df['primary_label'])]
more_data = more_data[more_data['file'].isin(og_df['filename'])]

# %%
# --------------- OLD COLUMNS ----------------
# 'primary_label',
# 'secondary_labels',
# 'type',
# 'latitude',
# 'longitude',
# 'scientific_name',
# 'common_name',
# 'author',
# 'license',
# 'rating',
# 'url',
# 'filename'

# --------------- NEW COLUMNS ----------------
# id', the XC id for the recording, also appears in file and filename
# 'gen', genus, taxonomy rank
# 'sp', species
# 'ssp', 84% nan, subspecies
# 'group', always says 'birds'
# 'en', english name
# 'rec', recorder, useless?
# 'cnt', country
# 'loc', city/location
# 'lat', latitude
# 'lng', longitude
# 'alt', elevation
# 'type', type of singing 'call' or 'song'
# 'sex', 75% nan, 14% uncertain
# 'stage', 77% nan, 12% adult
# 'method', field recording 99% or handheld, useless tbh?
# 'url', url to the recording, but without https: at the beginning, basically f'//xeno-canto.org/{id}'
# 'file', f'XC{id}'
# 'file-name', the actual name of the file, how it was uploaded
# 'sono', links to sonograms
# 'osci', links to oscillographs
# 'lic', license, some creative commons license
# 'q', the rating of the recording, A-E or 'no score'
# 'length', length of the recording
# 'time', time HH:MM of the recording
# 'date', date of the recording YYYY-MM-DD
# 'uploaded', date of the upload YYYY-MM-DD
# 'also', secondary label, 77% empty list,
# 'rmk', recording device
# 'bird-seen', bird seen or not 44% yes 34% no
# 'animal-seen', animal seen or not 44% yes 34% no, would guess the same as bird-seen
# 'playback-used', playback used or not 76% no 23% unknown (probably no i guess?)
# 'temp', null
# 'regnr',null
# 'auto', 92% no, 6% yes, automatic recording?
# 'dvc', device for recording, 87% null, diff from rmk, probably can be removed
# 'mic', 90% nan, microphone
# 'smp',
# 'primary_label', the primary label of the recording, the target
# 'numRecordings' number of recordings for the primary label

# %%
rename_cols = {
    'primary_label': 'primary_label',
    'also': 'secondary_labels',
    'type': 'type',
    'lat': 'latitude',
    'lng': 'longitude',
    # '': 'scientific_name', # MISSING?
    'en': 'common_name',
    'rec': 'author',
    'lic': 'license',
    'q': 'rating', # needs to be converted from E-A to 1-5
    'url': 'url',
    'file': 'filename' # convert to f'{primary_label}/{id}'
}
more_data['rating'] = more_data['q'].map({
    'E': 1,
    'D': 2,
    'C': 3,
    'B': 4,
    'A': 5
})
more_data['filename'] = more_data['primary_label'].str.cat(more_data['id'].astype(str), sep='/')
# %%
more_data = more_data.rename(columns=rename_cols)

# %%
# -------------- KEEP FROM NEW ----------------
# keep everything above, plus some extra columns
# TODO: finish this list and see what we need
new_cols = {
    'gen': 'genus',
    'sp': 'species',
    'cnt': 'country',
    'loc': 'loc',
}
more_data = more_data.rename(columns=new_cols)
more_data = more_data[[*rename_cols.values(), *new_cols.values()]]

more_data.to_csv('data/raw/train_data/train_meta_extended')

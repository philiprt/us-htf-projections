# ---------------------------------------------------------------------------

import pandas as pd
import pickle

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

url = 'https://tidesandcurrents.noaa.gov/publications/techrpt083.csv'

# ---------------------------------------------------------------------------

data = pd.read_csv(url, skiprows=15)

scn_split = data['Scenario'].str.split(pat=' - ', expand=True)
scn_split[0].loc[scn_split[0] == '0.3'] = 'low'
scn_split[0].loc[scn_split[0] == '0.5'] = 'int_low'
scn_split[0].loc[scn_split[0] == '1.0'] = 'int'
scn_split[0].loc[scn_split[0] == '1.5'] = 'int_high'
scn_split[0].loc[scn_split[0] == '2.0'] = 'high'
scn_split[0].loc[scn_split[0] == '2.5'] = 'extreme'
scn_split[1].loc[scn_split[1] == 'LOW'] = 17
scn_split[1].loc[scn_split[1] == 'MED'] = 50
scn_split[1].loc[scn_split[1] == 'HIGH'] = 83

data.loc[:, 'Scenario'] = scn_split[0]
data['Percentile'] = scn_split[1]

scn = {}

# ---------------------------------------------------------------------------

scn['meta'] = pd.DataFrame(data.iloc[:, [0, 2, 3]].values,
    index=data.iloc[:, 1], columns=['name', 'lat', 'lon'])
scn['meta'].index.name = 'PSMSL ID'
scn['meta'] = scn['meta'].groupby(scn['meta'].index).first()

# ---------------------------------------------------------------------------

scn['proj'] = pd.DataFrame(
    data.loc[:, 'RSL in 2000 (cm)':'RSL in 2200 (cm)'].values.T,
    index=list(range(2000,2101,10)) + [2120, 2150, 2200],
    columns=pd.MultiIndex.from_arrays(
        [data['PSMSL ID'], data['Scenario'], data['Percentile']]
        )
    )

# ---------------------------------------------------------------------------

fname = 'NOAA_SLR_scenarios.pickle'
with open(fname, 'wb') as f:
    pickle.dump(scn, f)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

url = 'https://tidesandcurrents.noaa.gov' + \
    '/publications/techrpt86_PaP_of_HTFlooding.csv'

cnt = {}

# ---------------------------------------------------------------------------

meta = pd.read_csv(url, skiprows=9, nrows=5)
cnt['meta'] = pd.DataFrame(meta.iloc[[0,1,2,4], 2:].values.T,
    index=meta.iloc[3, 2:], columns=['name', 'lat', 'lon', 'thrsh'])
cnt['meta'].index.name = 'noaa id'

# ---------------------------------------------------------------------------

cnt['proj'] = pd.read_csv(url, skiprows=15, header=None, index_col=[0, 1])
cnt['proj'].columns = cnt['meta'].index.values.tolist()
cnt['proj'].index.names = ['scenario', 'year']

# ---------------------------------------------------------------------------

cnt['cnvrt_id'] = pd.DataFrame([
        ['1611400', '058'],
        ['1612340', '057'],
        ['1612480', '061'],
        ['1615680', '059'],
        ['1617433', '552'],
        ['1617760', '060'],
        ['1619910', '050'],
        ['1630000', '053'],
        ['1770000', '056'],
        ['1820000', '055'],
        ['1890000', '051'],
        ['8413320', ''],
        ['8418150', '252'],
        ['8443970', '741'],
        ['8447930', '742'],
        ['8449130', '743'],
        ['8452660', '253'],
        ['8454000', ''],
        ['8461490', '744'],
        ['8467150', ''],
        ['8510560', '279'],
        ['8516945', ''],
        ['8518750', '745'],
        ['8519483', ''],
        ['8531680', ''],
        ['8534720', '264'],
        ['8536110', '746'],
        ['8545240', ''],
        ['8551910', ''],
        ['8557380', '747'],
        ['8571892', ''],
        ['8573364', ''],
        ['8574680', ''],
        ['8575512', ''],
        ['8577330', ''],
        ['8594900', ''],
        ['8631044', ''],
        ['8632200', ''],
        ['8635750', ''],
        ['8636580', ''],
        ['8638610', ''],
        ['8638863', '749'],
        ['8651370', '260'],
        ['8652587', ''],
        ['8656483', ''],
        ['8658120', '750'],
        ['8661070', ''],
        ['8665530', '261'],
        ['8670870', '752'],
        ['8720030', '240'],
        ['8720218', '753'],
        ['8721604', '774'],
        ['8723214', '755'],
        ['8723970', ''],
        ['8724580', '242'],
        ['8725110', '757'],
        ['8725520', ''],
        ['8726520', '759'],
        ['8726724', '773'],
        ['8727520', ''],
        ['8728690', '760'],
        ['8729108', ''],
        ['8729210', '761'],
        ['8729840', '762'],
        ['8735180', '763'],
        ['8747437', ''],
        ['8761724', '765'],
        ['8770570', '766'],
        ['8770613', ''],
        ['8771013', ''],
        ['8771450', '767'],
        ['8774770', '769'],
        ['8775870', '770'],
        ['8779770', '772'],
        ['9410170', '569'],
        ['9410230', '554'],
        ['9410660', '567'],
        ['9410840', '578'],
        ['9412110', '565'],
        ['9413450', '555'],
        ['9414290', '551'],
        ['9414750', ''],
        ['9415020', ''],
        ['9415144', ''],
        ['9416841', ''],
        ['9418767', ''],
        ['9431647', ''],
        ['9432780', ''],
        ['9435380', ''],
        ['9440910', ''],
        ['9444090', ''],
        ['9444900', ''],
        ['9447130', ''],
        ['9449424', ''],
        ['9449880', ''],
        ['9751401', ''],
        ['9751639', ''],
        ['9755371', ''],
        ['9759110', ''],
    ], columns=['noaa', 'uh'])

# ---------------------------------------------------------------------------

fname = 'NOAA_flood_counts.pickle'
with open(fname, 'wb') as f:
    pickle.dump(cnt, f)

# ---------------------------------------------------------------------------























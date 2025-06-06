import requests
import pandas as pd


def find_contract(tiker):
        tikers_list = []
        mount_list = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

        for i in range(0,7):
            for month in mount_list:
                new_tiker = tiker + month + str(i)
                tikers_list.append(new_tiker)
        print(tikers_list)
        return tikers_list



tikers = find_contract('GD')

flag_2 = 0
dataframes_list = []
flag = 0
for url in tikers:
    link = f'https://iss.moex.com/iss/engines/futures/markets/forts/boards/RFUD/securities/{url}/candles.json?interval=24'
    print(link)
    response = requests.get(link)
    # Считываем данные из json
    dt = response.json()
    if flag == 0:
        columns = dt['candles']['columns']
        row = dt['candles']['data']
        df = pd.DataFrame(row, columns = columns)
        df['SECID'] = url
        flag += 1
        dataframes_list.append(df)
    elif flag != 0:
        columns = dt['candles']['columns']
        row = dt['candles']['data']
        df = pd.DataFrame(row, columns = columns)
        df['SECID'] = url
        if flag_2 == 0:
            print(df)
            flag_2 = 1
        dataframes_list.append(df)
print(dataframes_list)
final_df = pd.concat(dataframes_list, sort = True)
final_df.to_excel(f'Gold.xlsx', index = False)


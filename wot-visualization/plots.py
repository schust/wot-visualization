import time
import re

from unidecode import unidecode
from main import get_db_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_player_winrates(db, ids, min_battles=100):
    sql = "SELECT accountid as player_id, CAST(wins AS float) / CAST(battlescount AS float) as player_winrate " \
          "FROM wot.player " \
          "WHERE accountid in %s and battlescount > %s" % (str(tuple(ids.values)), str(min_battles))  # that's a hack to get the proper string
    data = pd.read_sql(sql, con=db, index_col='player_id')
    return data


def get_tank_winrates(db, vehicle_id, min_battles=100):
    sql = "SELECT player_id, CAST(wins AS float) / CAST(battles AS float) as tank_winrate " \
          "FROM wot.player_vehicle " \
          "WHERE vehicle_id = %s and battles > %s" % (vehicle_id, str(min_battles))
    data = pd.read_sql(sql, con=db, index_col='player_id')
    return data


def process(db):
    tanks = get_tanks(db)
    tanks.sort_index(inplace=True, ascending=False)
    for tank_id, tank in tanks.iterrows():
        current_tank = 'tankid=[{0}]: {1}'.format(str(tank_id), tank['shortname'])
        print(current_tank)
        print(time.strftime('Current time: %H:%M:%S', time.localtime(time.time())))

        tank_winrates = get_tank_winrates(db, tank_id)
        if len(tank_winrates.index) < 100:  # todo maybe higher sample size? maybe add sample sizes to plot?
            print('Too few players({0}) with battles in tank:{1} found'.format(len(tank_winrates.index), tank_id))
            continue

        player_winrates = get_player_winrates(db, tank_winrates.index)

        merged = pd.concat([tank_winrates['tank_winrate'], player_winrates['player_winrate']], axis=1).reset_index(drop=True)
        merged['bins'] = pd.cut(merged['player_winrate'], bins=np.arange(0, 1, 0.005))

        binned = merged.groupby(merged['bins']).mean().dropna()  # todo count, and remove percents with < 100 players
        binned.index = binned.index.map(lambda x: x.mid)

        plot(binned, current_tank, tank_id, tank)


def plot(data, current_tank, tank_id, tank):
    plt.clf()
    plt.plot(list(data.index.values), data['tank_winrate'], linestyle='-', marker='.')
    axes = plt.gca()
    axes.set_xlim([0.4, 0.65])
    axes.set_ylim([0.4, 0.65])
    axes.set_xlabel('player winrate')
    axes.set_ylabel('tank winrate')
    axes.set_title(current_tank)
    axes.set_xticks(np.arange(0.4, 0.65, 0.05))
    axes.set_xticks(np.arange(0.4, 0.65, 0.01), minor=True)
    axes.set_yticks(np.arange(0.4, 0.65, 0.05))
    axes.set_yticks(np.arange(0.4, 0.65, 0.01), minor=True)
    plt.grid(b=True, which='both')
    axes.plot(axes.get_xlim(), axes.get_ylim(), ls="-", c=".3")
    # plt.show()
    plot_name = '{tier:02d}_{nation}_{tid}_{name}'.format(tier=tank['tier'], tid=tank_id, nation=tank['nation'], name=tank['shortname'])
    plot_name = unidecode(plot_name)
    plot_name = re.sub(r'\W+', '', plot_name)
    plt.savefig('plots/' + plot_name)


def get_tanks(db):
    return pd.read_sql("SELECT * from wot.vehicle", con=db, index_col='tankid')


def main():
    db = get_db_config()
    datapoints = process(db)


if __name__ == '__main__':
    main()

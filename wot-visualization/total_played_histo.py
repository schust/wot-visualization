import time
import re
import math

from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from unidecode import unidecode
from db import get_db_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_player_data(db, ids, min_battles=100):
    sql = "SELECT accountid as player_id, CAST(wins AS float) / CAST(battlescount AS float) as player_winrate " \
          "FROM wot.player " \
          "WHERE accountid in %s and battlescount > %s" % (
              str(tuple(ids.values)), min_battles)  # that's a hack to get the proper string
    data = pd.read_sql(sql, con=db, index_col='player_id')
    return data


def get_tank_winrates(db, vehicle_id, min_battles=100):
    sql = "SELECT player_id, CAST(wins AS float) / CAST(battles AS float) as tank_winrate, battles " \
          "FROM wot.player_vehicle " \
          "WHERE vehicle_id = %s and battles > %s" % (vehicle_id, str(min_battles))
    data = pd.read_sql(sql, con=db, index_col='player_id')
    return data


def total_played(db):
    tanks = get_tanks(db)
    tanks.sort_index(inplace=True, ascending=False)
    for tank_id, tank in tanks.iterrows():
        current_tank = 'tankid=[{0}]: {1}'.format(str(tank_id), tank['shortname'])
        print(current_tank)
        print(time.strftime('Current time: %H:%M:%S', time.localtime(time.time())))

        tanks = get_tank_winrates(db, tank_id)
        if len(tanks.index) < 100:  # todo maybe higher sample size?
            print('Too few players({0}) with battles in tank:{1} found'.format(len(tanks.index), tank_id))
            continue
        players = get_player_data(db, tanks.index)

        merged = pd.concat([tanks['battles'], tanks['tank_winrate'], players['player_winrate']], axis=1).reset_index(
            drop=True).dropna()
        merged['player_bins'] = pd.cut(merged['player_winrate'], bins=np.arange(0, 1, 0.005))
        merged['tank_bins'] = pd.cut(merged['tank_winrate'], bins=np.arange(0, 1, 0.005))

        # TODO maybe filter players with less than XX matches in tank to reduce outlier percentages
        player_sums = merged.groupby(merged['player_bins']).sum()
        player_sums['winrate'] = player_sums.index.map(lambda x: x.left)
        player_sums = player_sums[(player_sums['winrate'] >= 0.35) & (player_sums['winrate'] <= 0.65)].fillna(0)

        tank_sums = merged.groupby(merged['tank_bins']).sum()
        tank_sums['winrate'] = tank_sums.index.map(lambda x: x.left)
        tank_sums = tank_sums[(tank_sums['winrate'] >= 0.35) & (tank_sums['winrate'] <= 0.65)].fillna(0)

        total_matches = tank_sums.sum()[0].astype(int)

        plot(player_sums, tank_sums, current_tank, tank_id, tank, total_matches)


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot(player_data, tank_data, current_tank, tank_id, tank, total_matches):
    plt.clf()
    player_chart = plt.bar(player_data['winrate'], player_data['battles'], width=0.005, alpha=0.5)
    tank_chart = plt.bar(tank_data['winrate'], tank_data['battles'], width=0.005, alpha=0.5)

    axes = plt.gca()
    axes.set_xlabel('player winrate')
    axes.set_ylabel('battles played')
    axes.set_title('{tank}: {matches} matches'.format(tank=current_tank, matches=total_matches))
    axes.legend((player_chart[0], tank_chart[0]), ('Player Winrate', 'Tank Winrate'))

    plot_name = '{tier:02d}_{nation}_{tid}_{name}'.format(tier=tank['tier'], tid=tank_id, nation=tank['nation'],
                                                          name=tank['shortname'])
    plot_name = unidecode(plot_name)
    plot_name = re.sub(r'\W+', '', plot_name)
    plt.savefig('plots/total_played_histo/' + plot_name)
    # plt.show()


def get_tanks(db):
    return pd.read_sql("SELECT * FROM wot.vehicle", con=db, index_col='tankid')


if __name__ == '__main__':
    total_played(get_db_config())

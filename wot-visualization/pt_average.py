import time
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import PercentFormatter
from unidecode import unidecode
from db import get_db_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_player_winrates(db, ids, min_battles=100):
    sql = "SELECT accountid as player_id, CAST(wins AS float) / CAST(battlescount AS float) as player_winrate " \
          "FROM wot.player " \
          "WHERE accountid in %s and battlescount > %s" % (
          str(tuple(ids.values)), str(min_battles))  # that's a hack to get the proper string
    data = pd.read_sql(sql, con=db, index_col='player_id')
    return data


def get_tank_winrates(db, vehicle_id, min_battles=100):
    sql = "SELECT player_id, CAST(wins AS float) / CAST(battles AS float) as tank_winrate " \
          "FROM wot.player_vehicle " \
          "WHERE vehicle_id = %s and battles > %s" % (vehicle_id, str(min_battles))
    data = pd.read_sql(sql, con=db, index_col='player_id')
    return data


def pt_average(db):
    tanks = get_tanks(db)
    tanks.sort_index(inplace=True)
    for tank_id, tank in tanks.iterrows():
        current_tank = 'tankid=[{0}]: {1}'.format(str(tank_id), tank['shortname'])
        print(current_tank)
        print(time.strftime('Current time: %H:%M:%S', time.localtime(time.time())))

        tank_winrates = get_tank_winrates(db, tank_id)
        if len(tank_winrates.index) < 100:  # todo maybe higher sample size? maybe add sample sizes to plot?
            print('Too few players({0}) with battles in tank:{1} found'.format(len(tank_winrates.index), tank_id))
            continue

        player_winrates = get_player_winrates(db, tank_winrates.index)

        merged = pd.concat([tank_winrates['tank_winrate'], player_winrates['player_winrate']], axis=1).reset_index(
            drop=True)
        merged['bins'] = pd.cut(merged['player_winrate'], bins=np.arange(0, 1, 0.005))

        means = merged.groupby(merged['bins']).mean()
        counts = merged.groupby(merged['bins']).count()
        binned = means[counts > 10].dropna()  # -> only consider percentages with more than 10 players
        total_players = counts[counts > 10].sum()[0].astype(int)
        binned.index = binned.index.map(lambda x: x.mid)

        plot(binned, current_tank, tank_id, tank, total_players)


def plot(data, current_tank, tank_id, tank, total_players):
    plt.clf()
    plt.plot(list(data.index.values), data['tank_winrate'], linestyle='-', marker='.')
    axes = plt.gca()
    axes.set_xlabel('player winrate')
    axes.set_ylabel('tank winrate')
    axes.set_title('{tank}: {players} players'.format(tank=current_tank, players=total_players))

    axes.xaxis.set_major_formatter(PercentFormatter(1))
    axes.yaxis.set_major_formatter(PercentFormatter(1))

    lims = (0.4, 0.701)
    axes.set_xlim(lims)
    axes.set_ylim(lims)

    start, end = lims
    axes.xaxis.set_ticks(np.arange(start, end, 0.05), minor=False)
    axes.xaxis.set_ticks(np.arange(start, end, 0.025), minor=True)
    axes.yaxis.set_ticks(np.arange(start, end, 0.05), minor=False)
    axes.yaxis.set_ticks(np.arange(start, end, 0.025), minor=True)

    plt.grid(b=True, which='both')
    axes.plot((0.4, 0.7), (0.4, 0.7), ls="-", c=".3")
    plot_name = '{tier:02d}_{nation}_{tid}_{name}'.format(tier=tank['tier'], tid=tank_id, nation=tank['nation'],
                                                          name=tank['shortname'])
    plot_name = unidecode(plot_name)
    plot_name = re.sub(r'\W+', '', plot_name)
    plt.savefig('plots/pt_average/' + plot_name)


def get_tanks(db):
    return pd.read_sql("SELECT * FROM wot.vehicle", con=db, index_col='tankid')


if __name__ == '__main__':
    pt_average(get_db_config())

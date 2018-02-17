import json
import os

import pandas
import sqlalchemy as sql

def load_settings(path):
    with open(path) as json_data_file:
        settings = json.load(json_data_file)
        settings_dict = dict(settings)

    return settings_dict


def establish_db_connection(config):
    connection = 'postgresql+psycopg2://' + config['user'] + ':' + config['passwd'] + '@' + config['host'] + '/' + config['db']
    engine = sql.create_engine(connection)
    return engine


def get_db_config():
    fn = os.path.join(os.path.dirname(__file__), '..', 'res', 'settings.json')
    settings = load_settings(fn)
    postgres = settings['postgres']
    return establish_db_connection(postgres)


if __name__ == '__main__':
    db = get_db_config()

    # print(db.table_names('public'))

    # data = pandas.read_sql("SELECT COUNT(*) FROM public.player", con=db)
    data = pandas.read_sql("SELECT * FROM public.player_vehicle LIMIT 100", con=db)
    print(data)

    exit(0)

import json
import os

import sqlalchemy as sql

postgres = dict()
settings = dict()
settings_path = os.path.join(os.path.dirname(__file__), '..', 'res', 'settings.json')


def load_settings(path=settings_path):
    with open(path) as json_data_file:
        data = json.load(json_data_file)
        settings_dict = dict(data)

    print('Settings loaded.')

    return settings_dict


def establish_db_connection():
    connection = 'postgresql+psycopg2://' + postgres['user'] + ':' + postgres['passwd'] + '@' + postgres['host'] + '/' + postgres['db']
    engine = sql.create_engine(connection)
    print(engine)
    return engine


if __name__ == '__main__':
    fn = os.path.join(os.path.dirname(__file__), '..', 'res', 'settings.json')
    settings = load_settings()
    postgres = settings['postgres']
    db = establish_db_connection()

    exit(0)

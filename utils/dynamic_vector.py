import time
import redis
import json

env = 'STG'

redis_info = {'STG': {'url': 'rsq9dhwi-redisrep.dbsts.paic.com.cn', 'pwd': '@Wk6_oP9', 'port': 16907},
              'PRO': {'url': 'rsr3l9eg-redisrep.db.paic.com.cn', 'pwd': '@Wk6_oP9', 'port': 16794}}[env]
url, port, pwd = redis_info['url'], redis_info['port'], redis_info['pwd']

pool = redis.ConnectionPool(host=url, port=port, password=pwd)
db = redis.Redis(connection_pool=pool, decode_responses=True)
# slot_pattern = {'max_seconds': '', 'max_len': '', 'list': [], 'cache_time': ''}


class Redis_Update:
    def __init__(self):
        self.db = db
        self.key_pattern = "ailoanqc:vector:{}:{}"

    def get_context(self, request_id, name):
        slots = self.db.get(self.key_pattern.format(request_id, name))
        return json.loads(slots) if slots else None

    def update_context(self, request_id, name, key_dict):
        return self.db.set(self.key_pattern.format(request_id, name), json.dumps(key_dict), ex=key_dict['cache_time'])


class DynamicVector:
    def __init__(self):
        pass

    def verify_createdtime(self, info):
        max_seconds, info_list = info['max_seconds'], info['list']
        # No restrict in max_seconds<0
        if max_seconds >= 0:
            # Stack info: info time(timestamp)
            info_list = [i for i in info_list if time.time() - i[-1] <= max_seconds]
            info['list'] = info_list
        return info

    def verify_maxlength(self, info):
        info_list, max_len = info['list'], info['max_len']
        info_list = info_list[-max_len:]
        info['list'] = info_list
        return info

    def verify_normal(self, info):
        info = self.verify_createdtime(info)
        info = self.verify_maxlength(info)
        return info

    def get_list(self, record_id, name):
        """recordID  name"""
        info = ru.get_context(record_id, name)
        if info is None:
            return None
        info = self.verify_normal(info)
        return info['list']

    def insert_one(self, input_dict, record_id):
        """input_dict recordID name:longan/long/short max_seconds max_len cache_time"""
        long_info = ru.get_context(record_id, 'long')
        short_info = ru.get_context(record_id, 'short')

        if long_info is None:
            long_info = {'max_seconds': 10800, 'max_len': 100, 'list': [[input_dict, time.time()]], 'cache_time': 10800}
            _ = ru.update_context(record_id, 'long', long_info)
        else:
            # long: Update at least 10min
            if time.time() - long_info['list'][-1][-1] >= 600:
                long_info['list'].append([input_dict, time.time()])
                long_info = self.verify_normal(long_info)
                _ = ru.update_context(record_id, 'long', long_info)

        if short_info is None:
            short_info = {'max_seconds': 3600, 'max_len': 100, 'list': [[input_dict, time.time()]], 'cache_time': 3600}
            _ = ru.update_context(record_id, 'short', short_info)
        else:
            # short: Update at least 1min
            if time.time() - short_info['list'][-1][-1] >= 30:
                short_info['list'].append([input_dict, time.time()])
                short_info = self.verify_normal(short_info)
                _ = ru.update_context(record_id, 'short', short_info)

        return None


ru = Redis_Update()
dv = DynamicVector()
# c.insert_one('', '', '', 100, 10, 30)
# c.get_list('i','i')

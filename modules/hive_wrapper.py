from pyhive import hive

import pandas as pd

from traceback import print_exc

from typing import List, Type
from utils import foreach

##################################
# GLOBALS
# #################################

##########################
# CTQ = Create Table Query
# #########################

ALL_EDGES_CTQs = [
    """DROP TABLE IF EXISTS {}.all_edges""",

    """
        CREATE EXTERNAL TABLE {}.all_edges(
            stop_id STRING,
            next_stop_id STRING,
            stop_sequence INTEGER,            
            
            stop_name STRING,
            next_stop_name STRING,

            departure_time STRING,
            arrival_time STRING,
            next_arrival_time STRING,
            next_departure_time STRING,

            trip_id STRING,
            trip_headsign STRING,

            route_name STRING,
            route_type STRING,
            
            is_walkable BOOLEAN,

            duration_s DOUBLE,
            stop_waiting_time_s DOUBLE,

            node_id INTEGER,
            next_node_id INTEGER
        )
        STORED AS ORC
        LOCATION '/user/{}/network_data/all_edges'
    """
    ]

##################################
# CLASSES
# #################################

class HiveConnection:
    def __init__(self, username: str, host: str, port: str, location: str ="hive"):
        # database info
        self.name = username
        self.host = host
        self.port = port
        self.location = location if location is not None else f"{self.name}/{location}"
        # connection and cursor
        self.conn = hive.connect(host=self.host, port=self.port)
        self.cur  = self.conn.cursor()
        # cache
        self.cache = {}
        # setup database
        queries = [
            f"CREATE DATABASE IF NOT EXISTS {self.name} LOCATION '{self.location}'",
            f"USE {self.name}"
        ]
        foreach(self.cur.execute, queries)

    ###########################
    # SETUP DATABASE AND TABLES
    ###########################
      
    def create_tables(self, ctqs_list: List[str]) -> Type["HiveConnection"]:
        ctqs = [ctq.format(self.name) for ctqs in ctqs_list for ctq in ctqs]
        foreach(self.__exec, ctqs)
        return self
    
    def exec(self, query: str)-> None:
        assert self.conn is not None, "Connection is not established."
        assert self.cur  is not None, "Connection cursor is not established."
        
        try:
            return self.cur.execute(query)
        except Exception as e:
            self.close()
            print_exc()
        
        return None
    
    def pandas_df(self, query: str, force_reload: bool =False):
        if force_reload or (query not in self.cache.keys()):
            self.cache[query] = pd.read_sql(query, self.conn)       
        return self.cache[query]
    
    def purge_cache(self):
        self.cache = {}

    def reconnect(self) -> Type["HiveConnection"]:
        if self.conn is None:
            self.conn = hive.connect(host=self.host, port=self.port)
            self.cur  = self.conn.cursor()
        return self
    
    def close(self) -> None:
        self.conn.close()
        self.conn = None
        self.cur  = None
    

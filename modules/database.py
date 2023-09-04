import pandas as pd

class Database:
    def __init__(self, hive_conn):
        self.hive_conn = hive_conn
        self.cache = {}
        self.absolute_avg_delay = None
        self.absolute_mean_delay = None

    def __build_filename(self, **kwargs) -> pd.DataFrame:
        metric = kwargs.get("metric", None)
        day_period = kwargs.get("day_period", None)
        transport_type = kwargs.get("transport_type", None)
        transport_subtype = kwargs.get("transport_subtype", None)

        filename = metric + "_delay"
        filename += "" if transport_type is None else "_tsptype"
        filename += "" if transport_subtype is None else "_tspsubtype"
        filename += "" if day_period is None else "_dper"

        return filename

    def fetch_delay(self, metric: str, stop_name: str, force_fetch: bool =False, **kwargs) -> pd.DataFrame:
        transport_type = kwargs.get("transport_type", None)
        transport_subtype = kwargs.get("transport_subtype", None)
        day_period = kwargs.get("day_period", None)

        assert stop_name is not None, "stop_name is mandatory"

        assert metric is not None, "metric is mandatory"
        assert metric in ["avg", "std"], "metric must be one of 'avg', 'std'"
        
        assert day_period in ["morning", "prenoon", "afternoon", "latenoon", "evening", None], "day_period must be None or one of 'morning', 'prenoon', 'afternoon', 'latenoon', 'evening'"
        assert (transport_type is not None) or (transport_subtype is None), "transport_subtype is valid only if transport_type is specified"

        filename = self.__build_filename(metric=metric, **kwargs)
        if force_fetch or (filename not in self.cache.keys()):
            df = self.hive_conn.pandas_df(f"select * from {self.hive_conn.name}.{filename}")
            self.cache[filename] = df.rename(columns={name:name.split(".")[1] for name in df.columns})
            
        df = self.cache[filename]
        
        df = df[df["stop_name"] == stop_name]
        if len(df) == 0: return None

        if day_period is not None:
            df = df[df["day_period"] == day_period]
            if transport_type is not None:
                df = df[df["transport_type"] == transport_type]
                if len(df) == 0: return None
                if transport_subtype is not None:
                    df = df[df["transport_subtype"] == transport_subtype]
                    if len(df) == 0: return None

        assert len(df) == 1
        
        if metric == "avg":
            return float(df["mean_arrival_delay"].values[0])
        elif metric == "std":
            return float(df["std_arrival_delay"].values[0])
        else:
            raise NotImplemented()

    def purge_cache(self):
        self.cache = {}

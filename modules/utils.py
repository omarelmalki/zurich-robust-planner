import os
from IPython import get_ipython

def get_env_vars():
    username  = os.environ['USERNAME']
    host = os.environ['HIVE_SERVER2'].split(':')[0]
    port = os.environ['HIVE_SERVER2'].split(':')[1]
    return username, host, port

def setup_spark(username: str, server: str="http://iccluster044.iccluster.epfl.ch:8998") -> None:
    get_ipython().run_cell_magic(
        'spark',
        line='config', 
        cell="""{{ "name": "{0}-route-planner", "executorMemory": "4G", "executorCores": 4, "numExecutors": 10, "driverMemory": "4G" }}""".format(username)
    )
    get_ipython().run_line_magic("spark", f"""add -s {username}-route-planner -l python -u {server} -k""")

def foreach(f, xs):
    for x in xs: f(x)

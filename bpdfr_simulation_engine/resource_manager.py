import os

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.enhancement.roles import algorithm as roles_discovery

def find_roles_from_xes(file_path):
    log = xes_importer.apply(file_path)
    roles = roles_discovery.apply(log)
    for x in roles:
        print("%s -> %s" % (str(x[0]), str(x[1])))






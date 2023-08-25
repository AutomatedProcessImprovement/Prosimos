import csv
import re

stats_path = './assets/optimos_stats/experiment_stats/'
alloc_path = './assets/optimos_stats/explored_allocations/'
proc_names = ['LO-SL', 'LO-SH', 'LO-ML', 'LO-MH', 'P-EX', 'PRD', 'C-DM', 'INS', 'BPI-12', 'BPI-17']

variants = {'01': 'SKD',
            '02': 'RES',
            '03': 'SKD U RES',
            '04': 'SKD -> RES',
            '05': 'RES -> SKD'}


def main():
    for p_name in proc_names:
        for v in variants:
            cost_0, time_0 = read_csv_file(p_name, v)
            read_txt_file(p_name, v, cost_0, time_0)


def read_csv_file(p_name: str, v: str):
    with open("%s%s/%s.csv" % (alloc_path, p_name, v), 'r') as file:
        f_feader = csv.reader(file)
        is_next = False
        for row in f_feader:
            if is_next:
                return float(row[2]), float(row[3])
            if len(row) > 0 and row[0] == '# Iteration':
                is_next = True


def read_txt_file(p_name: str, v: str, cost_0: float, time_0: float):
    total = 0
    both_components = 0
    cost_component = 0
    time_component = 0
    with open("%s%s/%s.txt" % (stats_path, p_name, v), 'r') as file:
        for line in file.readlines():
            xx = re.findall(r'aCost: (\d+\.*\d*), cTime: (\d+):(\d+):(\d+\.*\d*)', line)
            yy = re.findall(r'aCost: (\d+\.*\d*), cTime: (\d+) days, (\d+):(\d+):(\d+\.*\d*)', line)
            if len(xx) > 0 or len(yy) > 0:
                xy = xx if len(xx) > 0 else yy
                total += 1
                cost = float(xy[0][0])
                time = float(xy[0][1]) * 3600 + float(xy[0][2]) * 60 + float(xy[0][3]) if len(xx) > 0 \
                    else float(xy[0][1]) * 86400 + float(xy[0][2]) * 3600 + float(xy[0][3]) * 60 + float(xy[0][4])
                if cost < cost_0 and time < time_0:
                    both_components += 1
                elif cost < cost_0:
                    cost_component += 1
                elif time < time_0:
                    time_component += 1

    print('%s - %s' % (p_name, variants[v]))
    print('Pareto Size: %d, Both: %.3f(%d), Time: %.3f(%d), Cost: %.3f(%d)' % (total,
                                                                               both_components / total, both_components,
                                                                               time_component / total, time_component,
                                                                               cost_component / total, cost_component))
    if both_components + time_component + cost_component != total:
        print("Wrong Sum")

    print('------------------------------------------------------------------------------')


if __name__ == "__main__":
    main()

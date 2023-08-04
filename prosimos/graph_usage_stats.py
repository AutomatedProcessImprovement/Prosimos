class GraphUsageStats:
    def __init__(self):
        self.data = dict()
        self.gateway_output_table_format = "| {:<15} | {:<50} | {:<25} | {:<10} | {:<50} |"

    def update_gateway(self, case_id, e_info, outgoing_arcs):
        element_id = e_info.id
        event_type = e_info.type

        if case_id not in self.data:
            self.data[case_id] = {}

        if element_id in self.data[case_id]:
            self.data[case_id][element_id]['usages'] += 1
            for arc in outgoing_arcs:
                arc_id = arc[0]
                if arc_id in self.data[case_id][element_id]['arc_usages']:
                    self.data[case_id][element_id]['arc_usages'][arc_id] += 1
                else:
                    self.data[case_id][element_id]['arc_usages'][arc_id] = 1
        else:
            self.data[case_id][element_id] = {
                "usages": 1,
                "event_type": event_type,
                "arc_usages": {}
            }
            for arc in outgoing_arcs:
                arc_id = arc[0]
                self.data[case_id][element_id]['arc_usages'][arc_id] = 1

    def print_gateways_as_table(self):
        print(self.gateway_output_table_format.format("Case_id", "Event_id", "Event_type", "Usages", "Arc_usages"))

        for case_id, case_data in self.data.items():
            for element_id, element_data in case_data.items():
                usages = element_data['usages']
                event_type = element_data['event_type']
                arc_usages = list(f"{arc_id}:{count}" for arc_id, count in element_data['arc_usages'].items())

                print(self.gateway_output_table_format.format(case_id, element_id, event_type, usages, arc_usages.pop(0) if arc_usages else ""))

                while arc_usages:
                    print(self.gateway_output_table_format.format("", "", "", "", arc_usages.pop(0)))

    def get_element_executions(self, case_id, event_id):
        if case_id in self.data and event_id in self.data[case_id]:
            return self.data[case_id][event_id]['usages']
        return 0

import json


class SimulationExecutionStats:
    def __init__(self):
        self.data = dict()
        self.stats = None

    def add_element(self, element_info, element_id):
        element_type = self._get_element_type(element_info)
        self.data[element_id] = {
            "BPMN_type": element_type,
            "name": element_info.name,
            "case_executions": {}
        }

    def _get_element_type(self, element_info):
        element_type = element_info.type.value
        return element_type if isinstance(element_type, str) else element_type[0]

    def update_element_execution(self, case_id, e_info, outgoing_arcs):
        element_id = e_info.id
        if case_id not in self.data[element_id]['case_executions']:
            self.data[element_id]['case_executions'][case_id] = {
                "executions": 1,
                "arcs": {}
            }
        else:
            self.data[element_id]['case_executions'][case_id]['executions'] += 1

        self._update_arcs(case_id, element_id, outgoing_arcs)

    def _update_arcs(self, case_id, element_id, outgoing_arcs):
        for arc in outgoing_arcs:
            arc_id = arc[0]
            arcs = self.data[element_id]['case_executions'][case_id]['arcs']
            arcs[arc_id] = arcs.get(arc_id, {"executions": 0})
            arcs[arc_id]['executions'] += 1

    def get_element_executions(self, case_id, event_id):
        return self.data.get(event_id, {}).get('case_executions', {}).get(case_id, {}).get('executions', 0)

    def _count_total_cases(self):
        return len(set(case for element in self.data.values() for case in element['case_executions']))

    def analyze_executions(self):
        total_cases = self._count_total_cases()
        self.stats = {
            'total_cases': total_cases,
            'elements': {element_id: self._analyze_element(element_id, data) for element_id, data in self.data.items()}
        }

    def _analyze_element(self, element_id, element_data):
        total_usages = sum(case_data['executions'] for case_data in element_data['case_executions'].values())
        arcs_info = self._collate_arcs_info(element_data)
        return {
            'id': element_id,
            'name': element_data['name'],
            'type': element_data['BPMN_type'],
            'total_usages': total_usages,
            'arcs': arcs_info
        }

    def _collate_arcs_info(self, element_data):
        arcs_info = {}
        for case_data in element_data['case_executions'].values():
            for arc_id, arc_data in case_data.get('arcs', {}).items():
                arcs_info[arc_id] = arcs_info.get(arc_id, 0) + arc_data['executions']
        return arcs_info

    def display_executions(self):
        if not self.stats:
            self.analyze_executions()
        self._display_element_stats()

    def _display_element_stats(self):
        total_cases = self.stats['total_cases']
        for element_id, element_data in self.stats['elements'].items():
            usage_percentage = (element_data['total_usages'] / total_cases) * 100 if total_cases else 0
            self._display_individual_element(element_id, element_data, total_cases, usage_percentage)
            print()

    def _display_individual_element(self, element_id, element_data, total_cases, usage_percentage):
        name, element_type = element_data['name'], element_data['type']
        if len(element_data['arcs']) > 1:
            print(f"{element_type} '{name}' (ID: {element_id}) has been used {element_data['total_usages']} times out of {total_cases} ({usage_percentage:.2f}%).")
            for arc_id, arc_usages in element_data['arcs'].items():
                arc_usage_percentage = (arc_usages / total_cases) * 100 if total_cases else 0
                print(f"\tFlow {arc_id} was executed {arc_usages} times out of {total_cases} ({arc_usage_percentage:.2f}%)")
        elif len(element_data['arcs']) == 1:
            arc_id, arc_usages = list(element_data['arcs'].items())[0]
            print(f"{element_type} '{name}' (ID: {element_id}) with flow {arc_id} has been used {element_data['total_usages']} times out of {total_cases} ({usage_percentage:.2f}%).")
        else:
            print(f"{element_type} '{name}' {element_type} (ID: {element_id}) has been used {element_data['total_usages']} times out of {total_cases} ({usage_percentage:.2f}%).")

    def find_issues(self):
        if not self.stats:
            self.analyze_executions()

        total_cases = self.stats['total_cases']
        warnings = []

        for element_id, element_data in self.stats['elements'].items():
            usage_percentage = (element_data['total_usages'] / total_cases) * 100 if total_cases else 0

            if usage_percentage < 1:
                warnings.append(
                    f"{element_data['type']} '{element_data['name']}' (ID: {element_id}) is used less frequently, only in {element_data['total_usages']} out of {total_cases} cases ({usage_percentage:.2f}%).")
            elif usage_percentage > 100:
                warnings.append(
                    f"{element_data['type']} '{element_data['name']}' (ID: {element_id}) is used more frequently than expected, in {element_data['total_usages']} out of {total_cases} cases ({usage_percentage:.2f}%).")

            for arc_id, arc_usages in element_data['arcs'].items():
                arc_usage_percentage = (arc_usages / total_cases) * 100 if total_cases else 0
                if arc_usage_percentage < 1:
                    warnings.append(
                        f"Flow {arc_id} from {element_data['type']} '{element_data['name']}' (ID: {element_id}) is used less frequently, only in {arc_usages} out of {total_cases} cases ({arc_usage_percentage:.2f}%).")
                elif arc_usage_percentage > 100:
                    warnings.append(
                        f"Flow {arc_id} from {element_data['type']} '{element_data['name']}' (ID: {element_id}) is used more frequently than expected, in {arc_usages} out of {total_cases} cases ({arc_usage_percentage:.2f}%).")

        return warnings

    def display_stats(self):
        if not self.stats:
            self.analyze_executions()
        print(json.dumps(self.stats, indent=4))

    def display_data(self):
        print(json.dumps(self.data, indent=4))


from prosimos.event_attributes import AllEventAttributes
from prosimos.case_attributes import AllCaseAttributes


class AllAttributes:
    def __init__(self, global_attributes, case_attributes, event_attributes):
        self.global_attributes = global_attributes

        global_attribute_names = {attr for attr in self.global_attributes.attributes}
        all_case_attribute_names = {attr.name for attr in case_attributes.attributes}
        all_event_attribute_names = set()
        for event in event_attributes.attributes:
            attrs = event_attributes.attributes[event]
            all_event_attribute_names.update(attrs.keys())

        global_case_attribute_names = global_attribute_names.intersection(all_case_attribute_names)
        global_event_attribute_names = global_attribute_names.intersection(all_event_attribute_names)

        self.global_case_attributes, self.case_attributes = \
            self._extract_case_attributes(case_attributes, global_case_attribute_names)

        self.global_event_attributes, self.event_attributes = \
            self._extract_event_attributes(event_attributes, global_event_attribute_names)

        local_case_attribute_names = all_case_attribute_names - global_case_attribute_names
        local_event_attribute_names = all_event_attribute_names - global_event_attribute_names
        attributes_intersection = local_case_attribute_names.intersection(local_event_attribute_names)
        if attributes_intersection:
            raise ValueError(f"Case attributes cannot be changed by event attributes. Check {attributes_intersection} attributes")

    def get_all_columns_generated(self):
        attributes_collections = [
            self.global_attributes,
            self.global_case_attributes,
            self.global_event_attributes,
            self.case_attributes,
            self.event_attributes
        ]
        columns = [attr.get_columns_generated() for attr in attributes_collections]
        return list(set(column for sublist in columns for column in sublist))

    def _extract_case_attributes(self, case_attributes, global_case_attribute_names):
        global_case_attributes = list()
        local_case_attributes = list()

        for attr in case_attributes.attributes:
            if attr.name in global_case_attribute_names:
                global_case_attributes.append(attr)
            else:
                local_case_attributes.append(attr)

        return AllCaseAttributes(global_case_attributes), AllCaseAttributes(local_case_attributes)

    def _extract_event_attributes(self, event_attributes, global_event_attribute_names):
        global_event_attributes = {}
        local_event_attributes = {}

        for event, attrs in event_attributes.attributes.items():
            for attr, value in attrs.items():
                if attr in global_event_attribute_names:
                    global_event_attributes.setdefault(event, {})[attr] = value
                else:
                    local_event_attributes.setdefault(event, {})[attr] = value
        return AllEventAttributes(global_event_attributes), AllEventAttributes(local_event_attributes)
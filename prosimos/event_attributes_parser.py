from prosimos.event_attributes import (AllEventAttributes, EventAttribute)


class EventAttributesParser:
    def __init__(self, json_data_with_event_attributes):
        self.data = json_data_with_event_attributes

    def parse(self):
        event_attributes = []
        for curr_event_attr in self.data:
            event_attr = EventAttribute(
                curr_event_attr["event_id"],
                curr_event_attr["name"],
                curr_event_attr["type"],
                curr_event_attr["values"]
            )
            event_attributes.append(event_attr)

        return AllEventAttributes(event_attributes)

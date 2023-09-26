from prosimos.event_attributes import (AllEventAttributes, EventAttribute)


class EventAttributesParser:
    def __init__(self, json_data_with_event_attributes):
        self.data = json_data_with_event_attributes

    def parse(self):
        event_attributes = {}
        for curr_event_attr in self.data:
            event_id = curr_event_attr["event_id"]
            event_attributes[event_id] = {}
            for attr in curr_event_attr["attributes"]:
                event_attr = EventAttribute(
                    event_id,
                    attr["name"],
                    attr["type"],
                    attr["values"]
                )
                event_attributes[event_id][attr["name"]] = event_attr

        return AllEventAttributes(event_attributes)
    

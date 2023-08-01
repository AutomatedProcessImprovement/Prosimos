from prosimos.global_attributes import (AllGlobalAttributes, GlobalAttribute)


class GlobalAttributesParser:
    def __init__(self, json_data_with_global_attributes):
        self.data = json_data_with_global_attributes

    def parse(self):
        global_attributes = {}
        for attr in self.data:
            global_attr = GlobalAttribute(
                attr["name"],
                attr["type"],
                attr["values"]
            )
            global_attributes[attr["name"]] = global_attr

        return AllGlobalAttributes(global_attributes)

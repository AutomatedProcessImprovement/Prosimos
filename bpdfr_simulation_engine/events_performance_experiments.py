import os
import string

from bpdfr_simulation_engine.control_flow_manager import (
    BPMN,
    EVENT_TYPE,
    BPMNGraph,
    ElementInfo,
)


def _add_events(bpmn_graph: BPMNGraph, sequence_flow_list, element_probability):
    # add events
    num_inserted_events = 5

    # keep track of the added events to the model
    inserted_events_logs: string = ""

    for i in range(num_inserted_events):
        # the flow arc to be replaced
        to_be_deleted_flow_arc = sequence_flow_list[i]
        to_be_deleted_flow_arc_id = to_be_deleted_flow_arc.attrib["id"]

        source_id = to_be_deleted_flow_arc.attrib["sourceRef"]
        target_id = to_be_deleted_flow_arc.attrib["targetRef"]

        new_event_id = f"event_{i}"

        # log info about the newly added event
        inserted_events_logs += f"{new_event_id} between {source_id} and {target_id}.\n"

        # add intermediate event
        bpmn_graph.add_bpmn_element(
            new_event_id,
            ElementInfo(
                BPMN.INTERMEDIATE_EVENT,
                new_event_id,
                new_event_id,
                EVENT_TYPE.TIMER,
            ),
        )

        # remove previously referenced sequence flow in the target activity
        bpmn_graph.remove_incoming_flow(target_id, to_be_deleted_flow_arc_id)

        # add sequence flow to and from the newly added event
        seq_flow_to_event = f"{source_id}_{new_event_id}"
        bpmn_graph.add_flow_arc(seq_flow_to_event, source_id, new_event_id)
        seq_flow_from_event = f"{new_event_id}_{target_id}"
        bpmn_graph.add_flow_arc(seq_flow_from_event, new_event_id, target_id)

        # duplicate the gateway probability if it existed before
        if source_id in element_probability:
            element_probability[source_id].update_candidate_key(
                to_be_deleted_flow_arc_id, seq_flow_to_event
            )

    # save logs about events
    logs_path = os.path.join(
        os.path.dirname(__file__),
        f"../performance_exp/events/input/events_logs_{num_inserted_events}.txt",
    )
    with open(logs_path, "w+") as logs_file:
        logs_file.write(inserted_events_logs)

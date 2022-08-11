# user-defined exceptions

class Error(Exception):
    """Base class for other exceptions"""
    pass


class InvalidBpmnModelException(Error):
    """Raised when the provided BPMN model is invalid"""


class InvalidLogFileException(Error):
    """Raised when the provided log file is invalid"""
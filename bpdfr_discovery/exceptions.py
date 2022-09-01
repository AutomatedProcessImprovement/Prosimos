# user-defined exceptions

class Error(Exception):
    """Base class for other exceptions"""
    pass


class NotXesFormatException(Error):
    """Raised when the provided filename is empty"""
    def __str__(self):
        return "Temporarily not supporting formats other than XES"


class InvalidInputDiscoveryParameters(Error):
    """Raised when the input parameters of to discover the simulation models are invalid"""

    def __str__(self):
        return "Invalid Input paramaters"

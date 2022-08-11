# user-defined exceptions

class Error(Exception):
    """Base class for other exceptions"""
    pass


class NotXesFormatException(Error):
    """Raised when the provided filename is empty"""
    def __str__(self):
        return "Temporarily not supporting formats other than XES"

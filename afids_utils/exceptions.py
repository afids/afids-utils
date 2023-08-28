"""Custom exceptions to raise for various errors"""


class InvalidFileError(Exception):
    """Exception raised when file to be parsed is invalid"""

    def __init__(self, message):
        super().__init__(message)


class InvalidFiducialError(Exception):
    """Exception for invalid fiducial selection"""

    def __init__(self, message):
        super().__init__(message)

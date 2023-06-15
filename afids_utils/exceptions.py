"""Custom exceptions"""


class InvalidFiducialNumberError(Exception):
    """Exception for invalid fiducial number"""

    def __init__(self, fid_num: int) -> None:
        super().__init__(f"Provided fiducial {fid_num} is not valid.")

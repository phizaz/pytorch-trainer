class GracefulException(Exception):
    pass


class TerminateLRException(GracefulException):
    pass

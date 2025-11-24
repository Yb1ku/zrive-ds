

msg = "[WARNING] This exception is not implemented yet. A normal Exception will be raised instead."


class UserNotFoundException(Exception):
    def __init__(self, message: str = msg):
        super().__init__(message)


class PredictionException(Exception):
    def __init__(self, message: str = msg):
        super().__init__(message)

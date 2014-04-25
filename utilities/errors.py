from runopts import DEBUG
class UserInputError(Exception):
    def __init__(self, message):
        if DEBUG:
            raise SyntaxError(message)
        raise SystemExit("UserInputError: {0}".format(message))

class WasatchError(Exception):
    def __init__(self, message):
        if DEBUG:
            raise
        raise SystemExit("WasatchError: {0}".format(message))

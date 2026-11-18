class APIError(Exception):
    def __init__(self, message, status=400, code="invalid_request_error", param=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.param = param

    def body(self):
        return {
            "error": {
                "message": self.message,
                "type": "server_error" if self.status >= 500 else "invalid_request_error",
                "param": self.param,
                "code": self.code,
            }
        }

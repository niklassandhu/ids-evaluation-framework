class NoDataLoaded(Exception):
    message = "No data was loaded. Please check the configuration and paths."

    def __init__(self, message):
        super().__init__(message)
        self.message = message

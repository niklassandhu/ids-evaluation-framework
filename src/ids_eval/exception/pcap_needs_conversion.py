class PcapNeedsConversion(Exception):
    message = "PCAP file format is not supported. Please convert to a supported format like CSV using tools like CICFlowMeter."

    def __init__(self, message):
        super().__init__(message)
        self.message = message

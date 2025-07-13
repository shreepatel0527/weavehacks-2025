class LabControlAgent:
    def __init__(self):
        self.instruments = {}

    def turn_on(self, instrument_name):
        self.instruments[instrument_name] = True
        print(f"{instrument_name} turned on.")

    def turn_off(self, instrument_name):
        if instrument_name in self.instruments:
            self.instruments[instrument_name] = False
            print(f"{instrument_name} turned off.")
        else:
            print(f"{instrument_name} is not currently on.")

    def is_on(self, instrument_name):
        return self.instruments.get(instrument_name, False)
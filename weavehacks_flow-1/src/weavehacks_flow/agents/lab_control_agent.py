import weave

class LabControlAgent:
    def __init__(self):
        self.instruments = {}

    @weave.op()
    def turn_on(self, instrument_name):
        self.instruments[instrument_name] = True
        print(f"{instrument_name} turned on.")
        weave.log({'instrument_control': {'action': 'turn_on', 'instrument': instrument_name}})

    @weave.op()
    def turn_off(self, instrument_name):
        if instrument_name in self.instruments:
            self.instruments[instrument_name] = False
            print(f"{instrument_name} turned off.")
            weave.log({'instrument_control': {'action': 'turn_off', 'instrument': instrument_name}})
        else:
            print(f"{instrument_name} is not currently on.")

    @weave.op()
    def is_on(self, instrument_name):
        status = self.instruments.get(instrument_name, False)
        weave.log({'instrument_status': {'instrument': instrument_name, 'is_on': status}})
        return status
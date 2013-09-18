
class BasalGanglia(object):

    def __init__(self, params):
        
        self.params = params

        self.create_networ()

    def get_eye_direction(self):
        pass


    def select_action(self, current_state):
        return current_state

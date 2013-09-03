
class BasalGanglia(object):

    def __init__(self, params):
        
        self.params = params


    def get_eye_direction(self):
        """
        Returns the efference copy, i.e. an internal copy of an outgoing (movement) signal.
        """
        pass


    def move_eye(self):
        """
        Select an action based on the current state and policy
        update the state
        """
        pass

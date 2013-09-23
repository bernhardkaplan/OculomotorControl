
class BasalGanglia(object):

    def __init__(self, params):
        
        self.params = params

#        self.create_network()

    def get_eye_direction(self):
        pass


    def select_action(self, current_state):
        action = [0., 0]

        action[0] = (current_state[0] - .5) * self.params['t_iteration'] / self.params['t_cross_visual_field'] # -.5 because the middle is .5
        action[1] = (current_state[1] - .5) * self.params['t_iteration'] / self.params['t_cross_visual_field'] # -.5 because the middle is .5

        over_react = 1.
        action[0] *= over_react
        action[1] *= over_react

#        action[0] = (current_state[0] - .5)# -.5 because the middle is .5
#        action[1] = (current_state[1] - .5)# -.5 because the middle is .5
        return action

    def create_network(self):
        pass

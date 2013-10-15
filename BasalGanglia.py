
class BasalGanglia(object):

    def __init__(self, params):
        
        self.params = params
        self.previous_action = [0., 0.]

#        self.create_network()

    def get_eye_direction(self):
        pass


    def select_action(self, current_state):
        """
        Return a tuple / list containing two elements:
         (v_x, v_y) -- meaning where to move the eye, i.e. how the image will be shifted
        """

        action = [0., 0.]

        # good tracking:
        action[0] = (current_state[0] - .5) + current_state[2] * self.params['t_iteration'] / self.params['t_cross_visual_field'] # -.5 because the middle is .5
        action[1] = (current_state[1] - .5) + current_state[3] * self.params['t_iteration'] / self.params['t_cross_visual_field'] # -.5 because the middle is .5

        # only following, no position correcton
#        action[0] = current_state[2] * self.params['t_iteration'] / self.params['t_cross_visual_field'] # -.5 because the middle is .5
#        action[1] = current_state[3] * self.params['t_iteration'] / self.params['t_cross_visual_field'] # -.5 because the middle is .5

        # disregarding the speed estimate
#        action[0] = (current_state[0] - .5)
#        action[1] = (current_state[1] - .5)

        self.previous_action = action
        return action
#        return [0, 0]

    def create_network(self):
        pass

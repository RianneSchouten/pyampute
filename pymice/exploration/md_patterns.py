"""Some title"""
# Author: Rianne Schouten <r.m.schouten@tue.nl>
# Co-Author: Davina Zamanzadeh <davzaman@gmail.com>
# Co-Author: 

class mdPattern():
    """
    some comments

    Parameters
    ----------
    What if there are no global parameters?
    """

    #def __init__(self):

    def _get_patterns(self, incomplete_data, print_statistics: True, print_plot: True):
        """Some comments"""

        s = 10
        self.patterns_statistics = s

        p = 10
        self.patterns_plot = p

        if print_statistics:
            print(self.patterns_statistics)
        if print_plot:
            print(self.patterns_plot)

        return self
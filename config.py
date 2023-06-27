import collections


class Config:

    def __init__(self):
        """

        """

        self.random_seed = 5

        self.DataCollection = collections.namedtuple(
            typename='DataCollection', field_names=['abscissae', 'ordinates', 'independent', 'dependent'])

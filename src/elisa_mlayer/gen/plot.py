import json

import numpy as np
from matplotlib import pyplot as plt


class Plot(object):

    @staticmethod
    def dataset(db_data_iterator, passband):

        f = plt.figure()
        ax = f.add_subplot(111)

        for batch in db_data_iterator():
            for record in batch:
                data = json.loads(record.data)
                plt.plot(data[0], np.array(data[1][passband]) / data[1][passband][70])

        ax.legend(loc=1)
        ax.set_xlabel("$Phase [-]$")
        ax.set_ylabel("$Flux [-]$")
        plt.show()

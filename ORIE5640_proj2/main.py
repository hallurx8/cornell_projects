if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import EM
    import matplotlib.pyplot as plt

    
    def generateModel(input):
        if input == 'ret':
            raw = pd.read_pickle('ret.pkl')
            init = 'vol'
        elif input == 'int':
            raw = pd.read_pickle('interest.pkl')
            init = 'trend'
        
        seq = np.array(raw)
        dates = raw.index

        mode = 2
        s_pred = EM.markow_switching_model(seq, mode, init)
        
        plt.plot(dates, s_pred)
        plt.show()


    generateModel('int')
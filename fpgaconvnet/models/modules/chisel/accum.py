import numpy as np

def utilisation_model(self):
    return {
        "LUT"   : np.array([self.filters,self.groups,
            self.data_width,self.cols,self.rows,self.channels]),
        "FF"    : np.array([self.filters,self.groups,
            self.data_width,self.cols,self.rows,self.channels]),
        "DSP"   : np.array([self.filters,self.groups,
            self.data_width,self.cols,self.rows,self.channels]),
        "BRAM"  : np.array([self.filters,self.groups,
            self.data_width,self.cols,self.rows,self.channels]),
    }



# Labels for the data.
class label():
    SEP = "." # Do not modify.

    # Physical data labels.
    freq = "freq"
    time = "time"
    acc  = "acc"
    vel  = "vel"
    pos  = "pos"
    temp = "temp"
    angle = "angle"

    # Suffixes.
    dft = "dft"     # discrete fourier transform
    psd = "psd"     # power spectral density
    ampl = "ampl"   # amplitude

    # Extract labels.
    def extract(self, s, depth=0):
        lst = s.split(self.SEP)
        if depth < 0:
            return lst[0]
        return self.SEP.join(lst[:depth+1])

    # Combine labels.
    def combine(self, *args):
        return self.SEP.join(list(args))
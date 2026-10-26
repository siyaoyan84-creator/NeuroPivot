"""NeuroPivot: real EEG representations and optional frozen AffectGPT conditioning."""
__version__="0.1.0"

def __getattr__(name):
    if name in {"EEGEncoder","AVConditioner","encode_eeg","condition_av"}:
        from neuropivot.inference import interfaces
        return getattr(interfaces,name)
    raise AttributeError(name)

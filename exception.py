class Error(Exception):
    pass


class MissingFileName(Error):
    pass


class MissingActivationArg(Error):
    pass


class MissingLossArg(Error):
    pass


class MissingOptimizerArg(Error):
    pass


class MissingMetricsArg(Error):
    pass


class EpochsNotANumber(Error):
    pass

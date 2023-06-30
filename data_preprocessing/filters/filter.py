class Filter:
    def __init__(self, **kwargs):
        self._flag = False
        self._output = dict()

    def filter_engine(self):
        pass

    def filter_output(self):
        return self._output

    @property
    def flag(self):
        return self._flag

    def filter(self):
        self.filter_engine()
        return self.filter_output()


class FilterPipeline(Filter):
    def __init__(self, filters, **kwargs):
        super(FilterPipeline, self).__init__()
        self._filters = filters
        self._kwargs = kwargs

    def filter_engine(self):
        for f in self._filters:
            self._kwargs.update(f(**self._kwargs).filter())

    def filter_output(self):
        return self._kwargs

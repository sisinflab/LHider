class GenericModule:
    def __init__(self, name='Generic Module', module_type='Module'):
        self._name = name
        self._type = module_type
        self.parameters = {}

    def __str__(self):
        params = "".join([f"\n\t{name}:\t{self.__getattribute__(attr)}" for name, attr in self.parameters.items()])
        if self.parameters:
            parameters = f'Parameters:{params}'
        else:
            parameters = ''
        name = f'Name:\t{self._name}'
        return f"{self._type} Information:\n" \
               f"{name}\n" \
               f"{parameters}\n"

    def __call__(self, *args, **kwargs):
        pass

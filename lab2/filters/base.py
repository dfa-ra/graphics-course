class FilterRegistry:
    _filters = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._filters[name] = func
            return func
        return decorator

    @classmethod
    def get_filters(cls):
        return cls._filters

    @classmethod
    def names(cls):
        return list(cls._filters.keys())

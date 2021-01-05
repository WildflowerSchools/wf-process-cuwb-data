from enum import Enum


class InteractionType(Enum):
    def __new__(cls, *args):
        if len(args) == 2:
            if isinstance(args, tuple):
                obj = object.__new__(cls)
                obj._value_ = args
                return obj

    @property
    def id(self):
        return self.value[0]

    @property
    def name(self):
        return self.value[1]

    @classmethod
    def as_id_name_dict(cls):
        return {c.id: c.name for c in cls}

    @classmethod
    def as_name_id_dict(cls):
        return {c.name: c.id for c in cls}

    @classmethod
    def as_id_list(cls):
        return [c.id for c in cls]

    @classmethod
    def as_name_list(cls):
        return [c.name for c in cls]

    CARRYING_FROM_SHELF = 0, 'CARRYING_FROM_SHELF'
    CARRYING_TO_SHELF = 1, 'CARRYING_TO_SHELF'
    CARRYING_FROM_AND_TO_SHELF = 2, 'CARRYING_FROM_AND_TO_SHELF'
    CARRYING_UNKNOWN = 3, 'CARRYING_UNKNOWN'
    NEXT_TO = 4, 'NEXT_TO'
    OTHER = 5, 'OTHER'


def interation_type_lookup(cls, value):
    keys = cls.__members__.keys()
    if isinstance(value, str):
        key = value.replace(' ', '_').upper()
        if key in keys:
            return getattr(cls, key)
    elif isinstance(value, int) and 0 <= value < len(keys):
        return list(cls.__members__.values())[value]
    else:
        return None


setattr(InteractionType, '__new__', interation_type_lookup)

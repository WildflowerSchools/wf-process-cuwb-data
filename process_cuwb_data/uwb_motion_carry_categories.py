from enum import Enum


class CarryCategory(Enum):
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

    NOT_CARRIED = 0, 'Not carried'
    CARRIED = 1, 'Carried'


def carry_category_lookup(cls, value):
    keys = cls.__members__.keys()
    if isinstance(value, str):
        key = value.replace(' ', '_').upper()
        if key in keys:
            return getattr(cls, key)
    elif isinstance(value, int) and 0 <= value < len(keys):
        return list(cls.__members__.values())[value]
    else:
        return None


setattr(CarryCategory, '__new__', carry_category_lookup)

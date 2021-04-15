from enum import Enum, EnumMeta

from nocasedict import NocaseDict


class CaseInsensitiveEnumMeta(EnumMeta):
    """
    Format lookup value before looking up enum
    """
    def __call__(cls, value, *args, **kw):
        # print("CaseInsensitiveEnumMeta:__call__: {} - {}".format(cls, value))
        keys = cls.__members__.keys()
        if isinstance(value, int) and value in cls.as_id_name_dict():
            value = cls.as_id_name_dict()[value]

        if isinstance(value, str):
            key = value.replace(' ', '_').upper()
            if key in keys:
                value = getattr(cls, key)

        return super().__call__(value, *args, **kw)


class CaseInsensitiveEnum(Enum, metaclass=CaseInsensitiveEnumMeta):
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
        return NocaseDict({c.name: c.id for c in cls})

    @classmethod
    def as_id_list(cls):
        return [c.id for c in cls]

    @classmethod
    def as_name_list(cls):
        return [c.name for c in cls]

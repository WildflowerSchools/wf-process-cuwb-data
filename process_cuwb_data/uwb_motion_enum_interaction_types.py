from process_cuwb_data.utils.case_insensitive_enum import CaseInsensitiveEnum


class InteractionType(CaseInsensitiveEnum):
    CARRYING_FROM_SHELF = 0, 'CARRYING_FROM_SHELF'
    CARRYING_TO_SHELF = 1, 'CARRYING_TO_SHELF'
    CARRYING_FROM_AND_TO_SHELF = 2, 'CARRYING_FROM_AND_TO_SHELF'
    CARRYING_UNKNOWN = 3, 'CARRYING_UNKNOWN'
    NEXT_TO = 4, 'NEXT_TO'
    OTHER = 5, 'OTHER'

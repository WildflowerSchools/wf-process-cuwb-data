from process_cuwb_data.utils.case_insensitive_enum import CaseInsensitiveEnum


class CarryCategory(CaseInsensitiveEnum):
    NOT_CARRIED = 0, 'Not carried'
    CARRIED = 1, 'Carried'

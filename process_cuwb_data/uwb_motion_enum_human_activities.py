from process_cuwb_data.utils.case_insensitive_enum import CaseInsensitiveEnum


class HumanActivity(CaseInsensitiveEnum):
    NOT_CARRYING = 0, 'Not carrying'
    CARRYING = 1, 'Carrying'

    # STANDING = 0, 'Standing'
    # CRAWLING = 1, 'Crawling'
    # WALKING = 2, 'Walking'
    # RUNNING = 3, 'Running'
    # SITTING_ON_CHAIR = 4, 'Sitting on Chair'
    # SITTING_ON_FLOOR = 5, 'Sitting on Floor'
    # LYING = 6, 'Lying'

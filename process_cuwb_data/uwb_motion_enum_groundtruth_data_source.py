from process_cuwb_data.utils.case_insensitive_enum import CaseInsensitiveEnum


class GroundtruthDataSource(CaseInsensitiveEnum):
    DATAPOINTS = 0, 'datapoints'
    IMU_TABLES = 1, 'imu_tables'

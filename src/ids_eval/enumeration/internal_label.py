from enum import Enum


class InternalLabel(Enum):
    TARGET_LABEL = "target_label"
    ATTACK_CATEGORY = "attack_category"

    @staticmethod
    def __values__() -> list[str]:
        return list(label.value for label in InternalLabel)

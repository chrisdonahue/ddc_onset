from enum import Enum

NUM_DIFFICULTIES = 5


class Difficulty(Enum):
    BEGINNER = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    CHALLENGE = 4


DIFFICULTY_TO_PLACEMENT_THRESHOLD = {
    Difficulty.BEGINNER: 0.15325437,
    Difficulty.EASY: 0.23268291,
    Difficulty.MEDIUM: 0.29456162,
    Difficulty.HARD: 0.29084727,
    Difficulty.CHALLENGE: 0.28875697,
}

from enum import Enum

SAMPLE_RATE = 44100
FRAME_RATE = 100


# Coarse difficulties from DDR. For general onset detection use, CHALLENGE is a good default.
class Difficulty(Enum):
    BEGINNER = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    CHALLENGE = 4


# Thresholds tuned on DDR validation data. Likely irrelevant for any other application.
DIFFICULTY_TO_PLACEMENT_THRESHOLD = {
    Difficulty.BEGINNER: 0.15325437,
    Difficulty.EASY: 0.23268291,
    Difficulty.MEDIUM: 0.29456162,
    Difficulty.HARD: 0.29084727,
    Difficulty.CHALLENGE: 0.28875697,
}

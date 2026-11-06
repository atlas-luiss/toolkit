"""NLP fine-tuning benchmarks: classification, QA, summarization, instruction tuning."""

from .classification import SST2, AGNews, MNLI
from .qa import SQuAD2
from .summarization import SAMSum, CNNDailyMail
from .instruction import Alpaca, GSM8K

__all__ = [
    "SST2", "AGNews", "MNLI",
    "SQuAD2",
    "SAMSum", "CNNDailyMail",
    "Alpaca", "GSM8K",
]

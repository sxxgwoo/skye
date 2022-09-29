from dataclasses import dataclass, field
from typing import List


@dataclass
class FAQDocumentItem:
    id: str = ''
    question: str = ''
    answer: str = ''


@dataclass
class STSRetrieveItem(FAQDocumentItem):
    score: float = 0.0


@dataclass
class TimeDocumentItem:
    time_zone: str = ''
    region: str = ''
    question: str = ''
    answer_type: str = ''

@dataclass
class STSTime(TimeDocumentItem):
    score: float = 0.0



@dataclass
class STSRetrieveResult:
    documents: List[STSRetrieveItem]
    msg: str = ''

@dataclass
class STSRetrieveResultTime:
    documents: List[TimeDocumentItem]
    msg: str = ''


# @dataclass 
# class sentimentDocumentItem:
#     positive: float = 0.0
#     neutral: float = 0.0
#     negative: float = 0.0



@dataclass
class HybridResult:
    bot_response_type: int = 0
    bot_response: str = ''
    # testtest: List[sentimentDocumentItem] = field(default_factory=list)
    sts_result: List[STSRetrieveItem] = field(default_factory=list)
    msg: str = ''
import typing
from dataclasses import asdict, dataclass, field


@dataclass
class Default:
    @property
    def __dict__(self):
        return asdict(self)


@dataclass
class UserTweet(Default):
    screen_name: str
    message: str
    reply:str


@dataclass
class RetrieverOutput(Default):
    query: typing.Optional[str] = None
    bm25_score: typing.Optional[float] = None
    db_name: typing.Optional[str] = None

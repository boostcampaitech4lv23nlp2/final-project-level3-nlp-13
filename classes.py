import typing
from dataclasses import asdict, dataclass, field


@dataclass
class Default:
    @property
    def __dict__(self):
        return asdict(self)


@dataclass
class UserTweet(Default):
    user_id: str
    screen_name: str
    message: str


@dataclass
class RetrieverOutput(Default):
    query: typing.Optional[str] = None
    bm25_score: typing.Optional[float] = None

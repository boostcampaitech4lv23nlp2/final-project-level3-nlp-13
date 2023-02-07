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
    user_name: str
    user_screen_name: str
    message: str
    tweet_id: str


@dataclass
class RetrieverOutput(Default):
    query: typing.Optional[str] = None
    bm25_score: typing.Optional[float] = None
    db_name: typing.Optional[str] = None

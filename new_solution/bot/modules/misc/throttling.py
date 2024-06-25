import asyncio
from aiogram import types, Dispatcher
from aiogram.dispatcher.handler import CancelHandler, current_handler
from aiogram.dispatcher.middlewares import BaseMiddleware
from aiogram.types import Message
from aiogram.utils.exceptions import Throttled

class ThrottlingMiddleware(BaseMiddleware):
    def __init__(self, limit=0.1, key_prefix='antiflood_') -> None:
        self.rate_limit = limit
        self.prefix = key_prefix
        super(ThrottlingMiddleware, self).__init__()

    async def on_process_message(self, message: Message, data: dict) -> None:
        handler = current_handler.get()
        dispatcher = Dispatcher.get_current()

        if handler:
            limit = getattr(handler, "throttling_rate_limit", self.rate_limit)
            key = getattr(handler, "throttling_key", f"{self.prefix}_{handler.__name__}")
        else:
            limit = self.rate_limit
            key = f"{self.prefix}_message"

        try:
            await dispatcher.throttle(key, rate=limit)
        except Throttled as t:
            await self.message_throttled(message, t, limit)
            raise CancelHandler() from t

    @staticmethod
    async def message_throttled(message: types.Message, throttled: Throttled, limit) -> None:
        if throttled.exceeded_count <= 2:
            await message.reply(f"<b>Пожалуйста, не спамьте. Подождите {limit} секунды до отправки следующего сообщения!</b>")


def rate_limit(limit: int, key=None):
    def decorator(func):
        setattr(func, "throttling_rate_limit", limit)
        if key:
            setattr(func, "throttling_key", key)
        return func

    return decorator
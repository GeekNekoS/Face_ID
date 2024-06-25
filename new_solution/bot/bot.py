import logging

from dotenv import load_dotenv # type: ignore
load_dotenv()

from aiogram.contrib.fsm_storage.memory import MemoryStorage
from modules.misc.album_middleware import AlbumMiddleware
from modules.misc.throttling import ThrottlingMiddleware
from aiogram import Dispatcher, Bot
from aiogram.utils import executor

from os import getenv, system
from modules.handlers import *


system('cls')
logging.basicConfig(level=logging.INFO)

bot = Bot(token=getenv('BOT_TOKEN'), parse_mode="HTML")
dp = Dispatcher(bot, storage=MemoryStorage())

if __name__ == '__main__':
    dp.middleware.setup(ThrottlingMiddleware())
    dp.middleware.setup(AlbumMiddleware())
    executor.start_polling(dp, skip_updates=True, on_startup=register_all_handlers(dp))

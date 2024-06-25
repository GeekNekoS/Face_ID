from aiogram import Dispatcher
from modules.handlers.user.main import register_user
from modules.handlers.admin.main import register_admin

def register_all_handlers(dp: Dispatcher) -> None:
    handlers = (
        register_admin,
        register_user
    )
    for handler in handlers:
        handler(dp)
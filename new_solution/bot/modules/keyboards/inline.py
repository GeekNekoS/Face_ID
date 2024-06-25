from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


authorization_menu = InlineKeyboardMarkup(row_width=1)\
    .add(
        InlineKeyboardButton(text="Авторизоваться", callback_data="user_auth"),
        # InlineKeyboardButton(text="Продолжить без авторизации", callback_data="user_without_auth")
        )

modeus_menu = InlineKeyboardMarkup(row_width=2)\
    .add(
        InlineKeyboardButton(text="🗓Расписание", callback_data="user_schedule"),
        InlineKeyboardButton(text="🧮БРС", callback_data="user_brs_marks"),
        )


week_schedule_menu = InlineKeyboardMarkup(row_width=2)\
    .add(
        InlineKeyboardButton(text="⬅️", callback_data="user_week_day_past"),
        InlineKeyboardButton(text="➡️", callback_data="user_week_day_next"),
        )

news_buttons_menu = InlineKeyboardMarkup(row_width=2)\
    .add(
        InlineKeyboardButton(text="⬅️", callback_data="user_news_past"),
        InlineKeyboardButton(text="➡️", callback_data="user_news_next"),
        )


settings_buttons = [
    InlineKeyboardButton(text="🥷Админка", callback_data="admin_main_menu"),
    InlineKeyboardButton(text="📄FAQ", callback_data="user_faq"),
    InlineKeyboardButton(text="👤Поддержка", callback_data="user_support"),
    InlineKeyboardButton(text="🚪Выйти с аккаунта", callback_data="user_logout_from_account"),
]


async def cancel_inline_menu(callback_data):
    return InlineKeyboardMarkup(row_width=1).add(InlineKeyboardButton(text="↪️Назад", callback_data=callback_data))






admin_menu =  InlineKeyboardMarkup(row_width=1)\
    .add(
        InlineKeyboardButton(text="📩Рассылка", callback_data="admin_message_distribution"),
        InlineKeyboardButton(text="💾Выгрузить базу пользователей", callback_data="admin_upload_database"),
        InlineKeyboardButton(text="🥷Добавить/удалить администратора", callback_data="admin_add_or_delete"),
    )
    
    
admin_confirm_distribution = InlineKeyboardMarkup(row_width=2)\
    .add(
        InlineKeyboardButton(text="Подтвердить", callback_data="admin_confirm_distribution"),
        InlineKeyboardButton(text="Отменить", callback_data="admin_cancel_distribution"),
    )
    
admin_add_delte_buttons = InlineKeyboardMarkup(row_width=2)\
    .add(
        InlineKeyboardButton(text="Добавить", callback_data="admin_add"),
        InlineKeyboardButton(text="Удалить", callback_data="admin_delete"),
    )
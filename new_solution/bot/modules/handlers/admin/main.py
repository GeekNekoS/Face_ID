from aiogram import Dispatcher
from aiogram.types import Message, CallbackQuery
from aiogram.dispatcher import FSMContext

from modules.keyboards import *
# from modules.misc.states import AdminMenuState


# async def get_users_statistics() -> None:
#     users_count = len(await get_id_users())
#     admins_username = '\n'.join(list(map(lambda username: f'<code>{username[0]}</code> @{username[1]}', await get_username_admins())))
#     return f'''
# 📊Статистика бота <code>
#  - 👤Пользоватлей: {users_count}
# </code>
# 🥷Администраторы
# {admins_username}
# '''

    
async def admin_handler(call: CallbackQuery) -> None:
    # if call.from_user.id in await get_id_admins():
    #     await call.message.edit_text(await get_users_statistics(), reply_markup=admin_menu)
    pass
    
    
# async def state_message_distribution(message: Message, state: FSMContext) -> None: 
#     await state.update_data(text = message.html_text)
#     await message.delete()
#     msg = (await state.get_data())['msg']
    
#     await msg.edit_text(f'Подтвердите отправку сообщения:\n\n {message.html_text}', reply_markup=admin_confirm_distribution)
#     await AdminMenuState.next()

            
# async def state_message_distribution_confirm(call: CallbackQuery, state:FSMContext) -> None:
#     text = (await state.get_data())['text']
#     msg = (await state.get_data())['msg']
    
#     match call.data:
#         case 'admin_confirm_distribution':
#             await msg.delete()
#             await send_message_for_all_users(text, call.from_user.id)
        
#         case 'admin_cancel_distribution':
#             await msg.edit_text(f'✅Вы отменили рассылку!')
    
#     await state.reset_data()        
#     await state.finish()
            
     
# async def state_admin_add_or_delete(message: Message, state: FSMContext) -> None:
#     user_data = message.text
#     msg = (await state.get_data())['msg']
#     state_data = (await state.get_data())['state_data']
#     dump_database = (await get_dump_database(user_data))
    
#     await message.delete()
    
#     if not dump_database:
#         await msg.edit_text(f'❌Пользователь <code>{message.text}</code> не найден!')
#         return await state.finish()
    
#     id, username, email = dump_database[0]
#     if state_data == 'add':
#         await add_admin_in_database({'id': id, 'username': username})
#         await msg.edit_text(f'✅Пользователь <code>{id if username == "None" else username}</code> успешно добавлен!')
#     else: 
#         await delete_admin_from_database({'id': id, 'username': username})
#         await msg.edit_text(f'✅Пользователь <code>{id if username == "None" else username}</code> успешно удалён!')
    
#     await state.finish()
  
            
# async def admin_callback_query_handler(call: CallbackQuery, state: FSMContext) -> None:
#     if call.from_user.id in await get_id_admins():
#         match call.data:
#             case 'admin_message_distribution':    
#                 await state.update_data(msg=await call.message.edit_text('Введите текст для рассылки: ', reply_markup=None))
#                 await AdminMenuState.users_message_distribution.set()
            
#             case 'admin_upload_database':
#                 users_count = len(await get_id_users())
#                 msg = f'👤Пользоватлей: {users_count}\n'
                
#                 for user in await get_dump_database():
#                     id, username, login = user
#                     msg += f'{id} | {username} | {login}\n'
                
#                 await call.message.edit_text(msg)
#                 await call.message.reply_document(open('src/studybot-telegram/data/users.db', 'rb'))
                
#             case 'admin_add_or_delete':
#                 await state.update_data(msg=await call.message.edit_text('Выберите действие', reply_markup=admin_add_delte_buttons))

#             case 'admin_add':
#                 await state.update_data(state_data='add')
#                 await call.message.edit_text('👤Введите id/username пользователя')
#                 await AdminMenuState.add_or_delete_admin.set()
        
#             case 'admin_delete':
#                 await state.update_data(state_data='delete')
#                 await call.message.edit_text('👤Введите id/username пользователя')
#                 await AdminMenuState.add_or_delete_admin.set()
                      
 
def register_admin(dp: Dispatcher) -> None:
    dp.register_callback_query_handler(admin_handler, lambda call: call.data == 'admin_main_menu')
    # dp.register_callback_query_handler(state_message_distribution_confirm, state=AdminMenuState.confirm_send_message_distribution)
    # dp.register_message_handler(state_admin_add_or_delete, state=AdminMenuState.add_or_delete_admin)
    # dp.register_message_handler(state_message_distribution, state=AdminMenuState.users_message_distribution)
    # dp.register_callback_query_handler(admin_callback_query_handler, lambda call: call.data.split('_')[0] == 'admin')
    

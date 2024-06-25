import asyncio
import os
import shutil
import cv2

from typing import List
from aiogram.types import Message
from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from EYWA import extract_faces, compare_faces, get_faces_vector
from modules.misc.states import UserInputState
from aiogram.types import Message
from modules.misc.throttling import rate_limit
from modules.keyboards import *
from modules.misc.settings import hamster_stickers
from random import choice


async def send_hello_message(message: Message) -> None:
    await message.answer(f'🦉<b>Добро пожаловать,</b> {message.from_user.first_name}! \
\nИзображения для теста можно взять отсюда — https://t.me/tests_photo', reply_markup=main_menu)
    
@rate_limit(2)
async def start_handler(message: Message) -> None:
    await message.answer_sticker(rf"{choice(hamster_stickers)}")
    await send_hello_message(message)


@rate_limit(2)
async def answer_user(message: Message):
    match message.text:
        case '✂️Обрезать лицо':
            await message.answer('🖼Пришлите изображение для обрезки', reply_markup=cancel_menu)
            await UserInputState.crop_image.set()
        case '⚖️Сравнить лица':
            await message.answer('👥Пришлите два изображения лица', reply_markup=cancel_menu)
            await UserInputState.compare_face.set()
        case _:
            await message.answer(f'🤖Вы вернулись в главное меню!', reply_markup=main_menu)
            
            
@rate_limit(0)
async def state_crop_image(message: Message, state: FSMContext, album: List[Message]=None) -> None: 
    if message.text == '↪️Назад':
        await message.answer(f'🤖Вы вернулись в главное меню!', reply_markup=main_menu)
        return await state.finish()
    
    match message.content_type:
        case 'photo' | 'document':
            await message.answer('✅Изображение получено, ожидайте...')
            if album is None: album = [ message ]
            for index, obj in enumerate(album):
                await asyncio.sleep(0.5)
                if obj.photo:
                    file_id = obj.photo[-1]
                else:
                    file_id = obj[obj.content_type]
                await file_id.download(destination_file=f'{message.from_user.id}/{message.from_user.id}_{index}.jpg')
        case _:
            await message.answer('⚠️Для обрезки лица небходимо прислать изображение!')
            return await UserInputState.crop_image.set()
            
    for image in os.listdir(f'{message.from_user.id}'):
        await asyncio.to_thread(extract_faces, f'{message.from_user.id}/'+image, 10, 512, f'{message.from_user.id}_results')
        
    if not os.path.exists(f'{message.from_user.id}_results'):
        await message.answer('⚠️Не удалось распознать лицо, повторите попытку!')
        return await UserInputState.crop_image.set()
   
    for image in os.listdir(f'{message.from_user.id}_results'):
        photo = open(f'{message.from_user.id}_results/{image}', 'rb')
        await message.answer_document(photo)

    await asyncio.to_thread(shutil.rmtree, f'{message.from_user.id}_results')
    await asyncio.to_thread(shutil.rmtree, f'{message.from_user.id}')
    await message.answer(f'✅Преобразование успешно!', reply_markup=main_menu)
    await state.finish()
    

@rate_limit(0)
async def state_compare_image(message: Message, state: FSMContext, album: List[Message]=None) -> None: 
    if message.text == '↪️Назад':
        await message.answer(f'🤖Вы вернулись в главное меню!', reply_markup=main_menu)
        return await state.finish()
    
    match message.content_type:
        case 'photo' | 'document':
            await message.answer('✅Изображение получено, ожидайте...')
            if album is None: album = [ message ]
            for index, obj in enumerate(album):
                await asyncio.sleep(0.5)
                if obj.photo:
                    file_id = obj.photo[-1]
                else:
                    file_id = obj[obj.content_type]
                await file_id.download(destination_file=f'{message.from_user.id}/{message.from_user.id}_{index}.jpg')
        case _:
            await message.answer('⚠️Для обрезки лица небходимо прислать изображение!')
            return await UserInputState.crop_image.set()

    faces = [cv2.imread(f'{message.from_user.id}/'+image) for image in os.listdir(f'{message.from_user.id}')]
    faces = await asyncio.to_thread(get_faces_vector, faces)
    result = await asyncio.to_thread(compare_faces, faces)
    await asyncio.to_thread(shutil.rmtree, f'{message.from_user.id}')
    await message.answer(f'Результат: <code>{round(result*100, 2)}%</code>', reply_markup=main_menu)
    await state.finish()
    
    
def register_user(dp: Dispatcher) -> None:
    dp.register_message_handler(start_handler, commands=['start'])
    
    dp.register_message_handler(
        state_crop_image,
        content_types=['photo', 'document', 'text'], 
        state=UserInputState.crop_image)
    
    dp.register_message_handler(
        state_compare_image,
        content_types=['photo', 'document', 'text'], 
        state=UserInputState.compare_face)

    dp.register_message_handler(answer_user, content_types=['text'])
    
    
    
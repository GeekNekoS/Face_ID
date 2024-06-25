from aiogram.dispatcher.filters.state import State, StatesGroup

class UserInputState(StatesGroup):
    crop_image = State()
    compare_face = State()
    

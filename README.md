![Python](https://img.shields.io/badge/-Python-05122A?style=flat&logo=python)&nbsp;

## *FACE ID*
Программа, помогающая считывать буквенные данные с фотографий
<br /> <br />


# Навигация
 - [Установка проекта на ПК](#download_project)
 - [Настройка готового проекта](#setting_up_a_project)
 - [Инструкция по установке Cuda](#cuda)
 - [Полезные команды](#useful_commands)
 - [Основные технологии / фрейморки](#basic_technologies)
 - [Полезная информация](#useful_information)
 - [Основные зависимости](#main_dependencies)
 - [Музычка для разработки](#nekos_music)
<br /> <br />


<a name="download_project"></a> 
## Установка проекта на ПК
1. Откройте консоль, вбив в поисковике ПК: <code>cmd</code>
2. Перейдите в директорию, куда хотите установить проект, пропишите следующую команду в консоль: <code>cd N:\Путь\до\папки\с\проектами</code>
3. Введите следующую команду: git clone https://github.com/[ник]/Face_ID.git
4. Откройте скачанный проект и можете приступать к разработке
<br /> <br />


<a name="setting_up_a_project"></a> 
## Настройка готового проекта
 - Версия Python: 3.11
 - После скачивания проекта к себе на компьютер не забудьте установить необходимые зависимости, прописав к консоли команду:  `pip install -r requirements.txt`
<br /> <br />


<a name="cuda"></a> 
## Инструкция по установке Cuda
1. Откройте PowerShell от имени администратора
2. Проверьте, что оказались в папке system32
3. Введите команду `wsl --install`
4. Подождите (долгую) установку ubuntu в windows 10-11
5. Зарегистрируйте Unix пользователя
6. Создайте venv через Ubuntu: https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#wsl-terminal
![img.png](data_for_readme/configuring_venv.png)
![img.png](data_for_readme/adding_interpriter.png)

7. Активируйте виртуальное окружение: `source venv/bin/activate`
8. Зарегистрируйтесь разработчиком Nvidia: https://developer.nvidia.com/developer-program
![img.png](data_for_readme/nvidia_dev_reg_example.png)
9. Скачайте и установите новый драйвер под свою видеокарту (проверить её характеристики можете следующим образом: диспетчер задач -> производительность -> Графический процессор Nvidia) с сайта Nvidia: https://www.nvidia.com/drivers
![img.png](data_for_readme/driver_choosing_example.png)
10. Скачайте и установите `cuda toolkit 11.8` для Вашей ОС Windows (там будет примерно 3 гига): https://developer.nvidia.com/cuda-11-8-0-download-archive

![img.png](data_for_readme/nvidia_toolkit_download_example.png)
11. Запустите check_video_cards_visibility.py, чтобы проверить видимость видеокарт, если не появляется сообщение `"Не вижу видеокарт!"`, шаг 12. можно пропустить
12. Скачайте и установите `cudnn 9.0`: https://developer.nvidia.com/cudnn-downloads
13. Обновите `pip`: `pip install --upgrade pip`
14. Установите `tensorflow` с `Cuda`: `pip install tensorflow[and-cuda]`
15. Установите Keras: `pip install --upgrade keras==3.0.0`
<br /> <br />


<a name="useful_commands"></a> 
## Полезные команды
 - Активация виртуального окружения (Ubuntu): `source venv/bin/activate`
 - Сохранить список зависимостей в файл requirements.txt: `pip freeze > requirements.txt`
 - Не забывайте активировать venv при работе через консоль Ubuntu: `source venv/bin/activate`
 - Отобразить зависимости в консоль: `pip list`
 - Удалить файл из контроля версий: `git rm --cache [путь до файла]`
 - Удалить все зависимости: `pip uninstall -y -r requirements.txt`
<br /> <br />


<a name="basic_technologies"></a> 
## Основные технологии / фрейморки
`OpenCV` / `Keras`, `Pillow`, `BytelIO`
<br /> <br />


<a name="useful_information"></a> 
## Полезная информация
 - [Пишем нейросеть на Python с нуля](https://proglib.io/p/pishem-neyroset-na-python-s-nulya-2020-10-07)
 - [Распознавание текста с изображения на Python | EasyOCR vs Tesseract | Компьютерное зрение](https://www.youtube.com/watch?v=H_nXZSM4WiU)
 - [Python + OpenCV + Keras: делаем распознавалку текста за полчаса](https://habr.com/ru/articles/466565/)
<br /> <br />


<a name="main_dependencies"></a> 
## Основные зависимости
1. Если работаете через Local terminal (Windows PowerShell) без Cuda:
`pip install tensorflow==2.15.0, keras==2.15.0, pillow==10.2.0`

2. Если работаете через Ubuntu terminal и установили Cuda:
`pip install tensorflow==2.15.0.post1, keras==3.0.0, pillow==10.2.0`
<br /> <br />


<a name="nekos_music"></a>
## Музычка для разработки [Neko's edition]
<details>
  <summary>
    <a href='https://www.youtube.com/watch?v=E_lpOJRIKgE'>Deep Drum and Bass Mix</a> ♡
  </summary>
  <img src='data_for_readme/deep_drum_and_bass_mix.png'></img>
</details> 

<details>
  <summary>
    <a href='https://www.youtube.com/watch?v=hbZ5UB_tM6o'>Don't You Stasis (V O E Remix)</a>
  </summary>
  <img src='data_for_readme/dont_you_stasis_voe_remix.png'></img>
</details> 

<details>
  <summary>
    <a href='https://www.youtube.com/watch?v=TwHS3c6zbwI'>Black Sun Empire: Podcast 04 HQ</a>
  </summary>
  <img src='data_for_readme/black_sun_empire_podcast_04_hq.png'></img>
</details> 

<details>
  <summary>
    <a href='https://www.youtube.com/watch?v=OoiP5mBRBwc'>Gothic Choirs and Ancient Cults</a>
  </summary>
  <img src='data_for_readme/gothic_choirs_and_ancient_cults.png'></img>
</details> 

<details>
  <summary>
    <a href='https://www.youtube.com/watch?v=ydHzwERb2rY&t=328s'>Rhan-Tegoth Part 1</a>, <a href='https://www.youtube.com/watch?v=ZGImCRim704'>Rhan-Tegoth Part 2</a>
  </summary>
  <img src='data_for_readme/rhan_tegoth.png'></img>
</details> 

<details>
  <summary>
    <a href='https://www.youtube.com/watch?v=56KHX6tMZ0g'>Celestial Warchants</a>
  </summary>
  <img src='data_for_readme/celestial_warchants.png'></img>
</details> 

<details>
  <summary>
    <a href='https://www.youtube.com/watch?v=000z5zd6mrc'>Shadowlands</a>
  </summary>
  <img src='data_for_readme/shadowlands.png'></img>
</details> 
<br /> <br />

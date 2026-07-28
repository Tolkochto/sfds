# Проект CV-4.  Перенос стиля на мобильном устройстве

### Описание проекта    
Cоздать правдоподобный прототип работающей модели переноса стиля.


### Какой кейс решаем?    
Создать мобильное приложение, реализующее функцию, а также демонстрирующую работу переноса стиля.


### Краткая информация о данных
Обучение ведется на основе случайных изображений и нескольких стилей.


### Этапы работы над проектом  
1. [Обучить модель переносу стиля](https://github.com/AlexexDenimus/sf_ds_projects/blob/master/cv_project_4/model/train.py)
2. [Перевести модель в tf lite](https://github.com/AlexexDenimus/sf_ds_projects/blob/master/cv_project_4/model/convert.py)
3. [Создать android приложение](https://github.com/AlexexDenimus/sf_ds_projects/tree/master/cv_project_4/app)
4. Добавить модель в мобильное приложения для переноса стиля

### Выводы:  
В ходе проекта было создано андроид приложение. Приложение позволяет пользователю сделать фото, применить к нему визуальный стиль и сохранить результат.

Фото обучения лежат в [папке](https://github.com/AlexexDenimus/sf_ds_projects/tree/master/cv_project_4/model/visualizations)

[Приложение](https://github.com/AlexexDenimus/sf_ds_projects/blob/master/cv_project_4/app-release.apk) 

[Пример работы приложения](https://github.com/AlexexDenimus/sf_ds_projects/blob/master/cv_project_4/example.mp4) 

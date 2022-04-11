# Intelligent placer #

**Большая лабораторная работа по курсу "Обработка и интерпретация сигналов"**

## Постановка задачи ##

По данной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику понимать, помещается ли каждый отдельный предмет в многоугольник. Предметы и горизонтальная поверхность, которые могут оказаться на фотографии, заранее известны. Также заранее известно направление вертикальной оси Z у этих предметов.

### Требования к фотографии предметов ###

Выбран следующий набор предметов:

+ Часы
+ Чехол для наушников
+ Маркер
+ Палетка
+ Чехол от очков
+ Ручка
+ Монета
+ Ключ
+ Кисть
+ Фигурка

Разрешение фотографии: 1920 х 1440 или 1472 х 1104

Высота съемки (расстояние до поверхности): 40 см

Предметы не могут перекрывать друг друга. Границы предметов не должны быть размыты (толщина линии границы не более 10px).

Фотографии исходных прдеметов и поверхности располагаются в папке data

### Требования к многоугольнику ###

Многоугольник задается фигурой, нарисованной темным маркером на белом листе бумаги, фотографируется вместе с предметами. 

Количество вершин многоугольника: 3 - 5 вершин.

## Сбор данных ##

Исходный дата сет и описание ожидаемых результатов работы алгоритма расположены по ссылке: https://drive.google.com/drive/folders/1tuVYLmBZEvJF36HdT3y9-wt4CrZfKBBY?usp=sharing

## Вывод программы ##

Исходное изображение, на котором отмечены границы фигуры и предмета и границы предмета в фигуре, если предмет можно в ней разместить

## План работы ##

#### Первая итерация ####
- Написать алгоритм распознавания границ объектов на изображении
- Разделить распознанные границы на границы фигуры и границы предметов
- Написать алгоритм переноса границ предмета в фигуру
- Протестировать алгоритм на нескольких изображений

#### Вторая итерация (возможная:)) ####
- Сделать новый датасет
- Добавить обработку ошибок
- Прогнать алгоритм для всех картинок в датасете
- Добавить метрики

## Разработчик ##

Дроздова Дарья, группа 5030102/80401 (GitHub: Drozdova-Daria, email: daria.1211@mail.ru)

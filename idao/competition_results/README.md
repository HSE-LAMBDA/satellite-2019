# Некоторые результаты и анализ IDAO 

### Структура

директории

* data – данные (скачать [тут](https://yadi.sk/d/1YjsfyXdAc6c1g))
   * данные с IDAO
   * данные, где зафиксированны все параметры КО, а коэффециент светового давления (КСД) и радиус перигея (РП) изменяется по сетке
* solutions – несколько лучших решений участников IDAO
* submission – прогнозы полученные с помощью алгоритмов участников  (скачать [тут](https://yadi.sk/d/xTfkF0KaWXx43w))

ноутбуки

* results_on_idao_data.ipynb – анализ результатов на данных IDAO 
    
    \+ сравнение с линейной регрессией 
    
    \+ сравнение с подходом, где предсказания SGP4 в test датасете обновляются на последней известной координате (перезапускается прогноз SGP4 и в качестве референсной точки используется последняя точка в train датасете)

* results_on_LPC_RP_data.ipynb – анализ результатов на данных, где зафиксированны все параметры КО, а коэффециент светового давления (КСД) и радиус перигея (РП) изменяются по сетке 

    \+ сравнение с линейной регрессией 

    \+ сравнение с подходом, где предсказания SGP4 в test датасете обновляются на последней известной координате (перезапускается прогноз SGP4 и в качестве референсной точки используется последняя точка в train датасете)


### Результаты

[на датасете IDAO](https://github.com/HSE-LAMBDA/satellite-2019/blob/master/idao/competition_results/results_on_idao_data.ipynb)

* SGP4

    * IDAO score:  85.35333249014583
    * SMAPE IDAO:  0.14646667509854172
    * SMAPE new:   0.22465428137020632

* Updated SGP4

    * IDAO score:  94.57101473641376
    * SMAPE IDAO:  0.05428985263586239
    * SMAPE new:   0.06542546046528024

* IDAO submisssion (data_o_plomo)

    * IDAO score:  97.21670571788155
    * SMAPE IDAO:  0.02783294282118438
    * SMAPE new:   0.056036577434143274

* IDAO submisssion (alsetboost)

    * IDAO score:  96.92031297743414
    * SMAPE IDAO:  0.030796870225658628
    * SMAPE new:   0.052050588061025874

[на датасете где все параметры зафиксированные, а КСД и Радиус Перигея изменяются по сетке](https://github.com/HSE-LAMBDA/satellite-2019/blob/master/idao/competition_results/results_on_LPC_RP_data.ipynb)

* SGP4

    * IDAO score:  37.324851755698354
    * SMAPE IDAO:  0.6267514824430165
    * SMAPE new:   0.39937790198419265

* Updated SGP4

    * IDAO score:  76.74469607670514
    * SMAPE IDAO:  0.23255303923294857
    * SMAPE new:   0.17122228662490271

* IDAO submisssion (data_o_plomo)

    * IDAO score:  84.78633720366511
    * SMAPE IDAO:  0.15213662796334887
    * SMAPE new:   0.12034785153708875

# Conclusions

* Using "Updated SGP4" doesn't improve the models' score (?!)
* High error at low Perigee Radius
* IDAO models do quite well
* To going deeper see graphs here
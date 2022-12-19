# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 08:34:31 2022

@author: raugom
"""

import influxdb_client
from datetime import datetime,timedelta
import math
import numpy as np
from random import randrange
import collections
import operator
import os
        
bucket = "----- BUCKET OF INFLUX-DB -----"
org = "----- ORG OF INFLUX-DB -----"
token = "----- TOKEN OF INFLUX-DB -----"
# Store the URL of your InfluxDB instance
url="----- URL OF INFLUX-DB -----"

client = influxdb_client.InfluxDBClient(
   url=url,
   token=token,
   org=org
)
query_api = client.query_api()

actividades = dict()
actividades = {
    "BATHROOM ACTIVITY" : 0, "CHORES" : 1, "COOK" : 2, "DISHWASHING" : 3, "DRESS" : 4,
    "EAT" : 5, "LAUNDRY" : 6, "MAKE SIMPLE FOOD" : 7, "OUT HOME" : 8, "PET" : 9,
    "READ": 10, "RELAX" : 11, "SHOWER" : 12, "SLEEP" : 13, "TAKE MEDS" : 14, "WATCH TV" : 15, 
    "WORK" : 16, "OTHER" : 17
}

all_sensors = [
                ["Vibration","v",11,"_action",["state"]],
                ["TempHum","th",2,"_temperature",["temperature","humidity"]],
                ["Contact","c",8,"_contact",["state"]],
                ["Movement","m",8,"_occupancy",["state"]],
                ["Illuminance","l",2,"_illuminance_lux",["illuminance_lux"]],
                ["Power","p",2,"_power",["power"]]
              ]

all_beacons = [["user1",["state","distance"]]]
ID_beacons = ["banho","cocina","dormitorio","estudio","not_home","pasillo","salon1"]

share_activities = ["CHORES","COOK","PET","READ"]

filtros = dict()
filtros = {
    "OTHER" : 0.05,
    "SLEEP" : 0.05,
    "OUT HOME" : 0.05,
    "WATCH TV" : 0.6
}

#RUTA GUARDADO
save_path = './npy/splitted/Prueba/User2/'

# RUTA COMPARTIR
share_path = './npy/splitted/Prueba/User1/'

# PORCENTAJES ENTRENAMIENTO
validation_percent = 0.1
test_percent = 0.2

segundosDIVISION = 2
tamanoVENTANA = 60
max_lenght = 35
stack_size = 2
date_ini = "2022-04-23T00:00:00.000Z"
date_end = "2022-06-23T23:59:59.999Z"

str_time = 'from(bucket: "' + bucket + '")\
              |> range(start: ' + date_ini + ', stop: ' + date_end + ')'
        
def sortByDate(elem):
    return datetime.strptime(elem[0], '%Y-%m-%dT%H:%M:%S.%fZ')

def date_generator(row):
    fecha = row[0]
    fecha = datetime.strptime(fecha,"%Y-%m-%dT%H:%M:%S.%fZ")
    return fecha

def getList(dict):
    return list(dict.keys())

def get_key(my_dict,val):
    for key, value in my_dict.items():
         if val == value:
             return key
 
    return "key doesn't exist"
def row_generator(sensores, actividades, ACT, horaX, horaY):
     
    # HORA + SENSORES + BEACONS
    row = [horaX, horaY, sensores["c1"], sensores["c2"], sensores["c3"], sensores["c4"], sensores["c5"], 
           sensores["c6"], sensores["c7"], sensores["c8"], sensores["h1"], sensores["h2"], sensores["l1"],
           sensores["l2"], sensores["m1"], sensores["m2"], sensores["m3"], sensores["m4"], sensores["m5"], 
           sensores["m6"], sensores["m7"], sensores["m8"], sensores["p1"], sensores["p2"], sensores["t1"],
           sensores["t2"], sensores["v1"], sensores["v2"], sensores["v3"], sensores["v4"], sensores["v5"],
           sensores["v6"], sensores["v7"], sensores["v8"], sensores["v9"], sensores["v10"], sensores["v11"],
           sensores["banho"], sensores["cocina"], sensores["dormitorio"], sensores["estudio"],
           sensores["not_home"], sensores["pasillo"], sensores["salon1"],
           actividades[ACT]]
           
    return row

def generate_sin_cos_hour(hora_min):
    
    hora = int(hora_min[0:2])
    minuto = int(hora_min[3:5])
    segundo = int(hora_min[6:8])
    horaX = math.cos((2*math.pi*(hora + (minuto/60) + (segundo/3600)))/(24))
    horaY = math.sin((2*math.pi*(hora + (minuto/60) + (segundo/3600)))/(24))
    
    #                                       (x) - minimo
    # NORMALIZO LOS VALORES ENTRE 0 Y 1 ->  -------------
    #                                      maximo - minimo
    
    horaX = (horaX + 1)/(2)
    horaY = (horaY + 1)/(2)
    
    return horaX, horaY

def generate_empty_activity_dict(array):
    keys = getList(actividades)
    keys = np.arange(0,len(keys),1,dtype=int)
    if(array == True):
        values = []
        for activity in range(0,len(keys)):
            values.append([])
    else:
        values = np.zeros(len(keys),dtype=int)
    new_dict = dict(zip(keys,values))
    return new_dict

def match_dictionaries(dict_base, dict_final):
    for key in dict_base.keys():
        dict_final[key] = dict_base[key]
    return dict_final
       
def read_data(sensores,sensors_list):
    for sensor_type in all_sensors:
        for num in range(1,sensor_type[2]+1):
            if(sensor_type[1]=="th"):
                sensor = "t" + str(num) + "h" + str(num)
            else:
                sensor = sensor_type[1] + str(num)
            for campo in sensor_type[4]:
                if(campo == "temperature"):
                    ref = "t" + str(num)
                elif(campo == "humidity"):
                    ref = "h" + str(num)
                else:
                    ref = sensor
                sensors_list.append(ref)
                query_data = str_time + '\
                  |> filter(fn: (r) => r["entity_id"] == "' + sensor + sensor_type[3] + '")\
                  |> filter(fn: (r) => r["_field"] == "' + campo + '")'
                result = query_api.query(org=org, query=query_data)
                for table in result:
                  for record in table.records:
                      regist = []
                      regist.append(record.get_time().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
                      regist.append(ref)
                      regist.append(record.get_value())
                      sensores.append(regist)
                      

    for user_presence in all_beacons:
        beacons = []
        sensor = user_presence[0] + "_presence"
        for field in user_presence[1]:      
            query_data = str_time + '\
                  |> filter(fn: (r) => r["entity_id"] == "' + sensor + '")\
                  |> filter(fn: (r) => r["_field"] == "' + field + '")'
            result = query_api.query(org=org, query=query_data)
            for table in result:
                field = []
                for record in table.records:
                    regist = []
                    regist.append(record.get_time().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
                    regist.append(record.get_value())
                    field.append(regist)
            beacons.append(field)
        distances = dict(beacons[1])
        for line in beacons[0]:
            if(line[0] in distances):
                regist = []
                regist.append(line[0])
                regist.append(line[1])
                regist.append(distances[line[0]])
                sensores.append(regist)
                if line[1] not in sensors_list:
                    sensors_list.append(line[1])

    sensores.sort(key=sortByDate)
    
    return sensores

def read_activities(activity):
    activities = []
    query_data = str_time + '\
          |> filter(fn: (r) => r["entity_id"] == "' + activity + '")\
          |> filter(fn: (r) => r["_field"] == "state")\
          |> filter(fn: (r) => r["source"] == "HA")'
    result = query_api.query(org=org, query=query_data)
    for table in result:
      for record in table.records:
          regist = []
          valor = ""
          regist.append(record.get_time().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
          if(record.get_value()=="MEDITATE"):
              valor = "RELAX"
          else:
              valor = record.get_value()
          regist.append(valor)
          activities.append(regist)
    
    return activities

def get_statistics(sensores,sensors_list):
    dicStatistics = dict()
    for regist in sensores:        
        if(regist[2] == "on" or regist[2] == "vibration" or regist[2] == "tilt" or regist[2] == "drop"): 
            regist[2] = 1            
        elif(regist[2] == "off" or regist[2] == "None"): 
            regist[2] = 0           
        else:
            regist[2] = float(regist[2])        
        if regist[1] in dicStatistics.keys(): # SI EXISTEN ESTADISTICAS DEL SENSOR
            if(regist[2] < dicStatistics[regist[1]][0]): # NUEVO MINIMO
                dicStatistics[regist[1]][0] = regist[2]
            if(regist[2] > dicStatistics[regist[1]][1]): # NUEVO MAXIMO
                dicStatistics[regist[1]][1] = regist[2]
        else: # SI NO EXISTEN ESTADISTICAS DEL SENSOR
            statistic = [regist[2],regist[2]]
            dicStatistics[regist[1]] = statistic  
    return dicStatistics

def load_dictionary(sensors_list,dicStatistics):
    dictionary = dict()
    for key in sensors_list:
        dictionary[key] = dicStatistics[key][0]
    return dictionary

def normalize(dicSensoresAcum,dicStatistics):
    for key in dicSensoresAcum.keys():
        if((dicStatistics[key][1]-dicStatistics[key][0]) != 0 and key != 'p2'):
            if(len(key) <= 3): # SI ES UN SENSOR
                dicSensoresAcum[key] = (dicSensoresAcum[key]-dicStatistics[key][0])/(dicStatistics[key][1]-dicStatistics[key][0])   
            else: # SI ES UN BEACON
                if(dicSensoresAcum[key]>dicStatistics[key][0]):
                    dicSensoresAcum[key] = 1-((dicSensoresAcum[key]-dicStatistics[key][0])/(dicStatistics[key][1]-dicStatistics[key][0]))
                else:
                    dicSensoresAcum[key] = 0
            
    return dicSensoresAcum

def load_dataset(sensores,date_ini,date_end,dicStatistics):
    
    sensores_procesados = []
    p2_ant = 0
    p2_umbral = 5
    p2_counter = 150
    
    contador = 0
    contador_act = 0
    num_filas = 0
    ACT = "OTHER"
    date_ini = datetime.strptime(date_ini,"%Y-%m-%dT%H:%M:%S.%fZ")
    date_end = datetime.strptime(date_end,"%Y-%m-%dT%H:%M:%S.%fZ")
    fecha = date_ini - timedelta(seconds=date_ini.second)
    
    dicSensores = load_dictionary(sensors_list,dicStatistics)
    dicSensoresAcum = load_dictionary(sensors_list,dicStatistics)
    dicSensoresAcum = match_dictionaries(dicSensores,dicSensoresAcum)
    stop = False
    while(fecha < date_end):
        # SI OCURRE UN EVENTO EN EL FICHERO DE DATOS
        while(fecha > date_generator(sensores[contador]) and stop == False):
            # RECONOZCO EL SENSOR
            ID_sensor = sensores[contador][1]
            
            # RECONOZCO EL ESTADO
            ID_estado = sensores[contador][2]
            
            if(ID_estado == "on" or ID_estado == "vibration" or ID_estado == "tilt" or ID_estado == "drop"): 
                ID_estado = 1
                dicSensoresAcum[ID_sensor] = ID_estado # ME QUEDO CON EL VALOR 1 AUNQUE SEA UN VALOR ESPORADICO
                
            elif(ID_estado == "off" or ID_estado == "None"): 
                ID_estado = 0
                
            elif(ID_sensor == 'p2'):
                valor = ID_estado
                if(ID_estado > p2_umbral and p2_ant < p2_umbral): # SOBREPASO DE UMBRAL
                    ID_estado = 1
                    p2_counter = 0
                elif(p2_counter<150):
                    ID_estado = 1
                else:
                    ID_estado = 0
                p2_ant = valor
                dicSensoresAcum[ID_sensor] = ID_estado
                
            else:
                ID_estado = float(ID_estado)
                dicSensoresAcum[ID_sensor] = ID_estado
            
            if(p2_counter>150):
                dicSensoresAcum['p2'] = 0
            else:
                p2_counter += 1
            
            if ID_sensor in ID_beacons: # SI SE TRATA DE UN BEACON
                for beacon in ID_beacons:
                    dicSensores[beacon] = 0                              
                
            # ACTUALIZO EL DICCIONARIO
            dicSensores[ID_sensor] = ID_estado # AQUI SE GUARDA EL VALOR REAL EN CADA MOMENTO            
            
            if(contador_act < len(activities)):
                # SI OCURRE UN CAMBIO EN LA ACTIVIDAD
                if(fecha > date_generator(activities[contador_act])):
                    
                    #ACT_ANT = ACT_NUEVA
                    ACT = activities[contador_act][1]
                    contador_act += 1
                    #matrizTrans[actividades[ACT_ANT]][actividades[ACT_NUEVA]] += 1
            
            if(contador >= len(sensores)-1):
                stop=True
            else:
                contador += 1
            
        fecha_cat = datetime.strftime(fecha,"%Y-%m-%dT%H:%M:%S.%fZ")[11:19]
        horaX, horaY = generate_sin_cos_hour(fecha_cat)
        dicSensoresAcum = normalize(dicSensoresAcum,dicStatistics)
        sensores_procesados.append(row_generator(dicSensoresAcum, actividades, ACT, horaX, horaY))
        dicSensoresAcum = match_dictionaries(dicSensores,dicSensoresAcum)
        num_filas += 1
        fecha = fecha + timedelta(seconds=segundosDIVISION)

    sensores_procesados = np.array(sensores_procesados,dtype=np.float32)
    return sensores_procesados

def convert_proccesed_rows(filas_procesadas):

    T = filas_procesadas[:,0:2]
    Xventana = filas_procesadas[:,2:44]
    Yventana = filas_procesadas[:,44]
        
    return Xventana, Yventana, T
    
def sliding_window(Xventana, Yventana, T):
    
    Yventana = Yventana[tamanoVENTANA-1:]
    T = T[tamanoVENTANA-1:]
    Xventana = np.lib.stride_tricks.sliding_window_view(Xventana,(tamanoVENTANA,42)) 
    Xventana = np.reshape(Xventana,(Xventana.shape[0],tamanoVENTANA*42))
    Xventana = np.append(np.array(T),np.array(Xventana),axis=1)
    return Xventana, Yventana

def generate_train_test_validation(days, test_percent, validation_percent):
    
    Ntest = round(len(days)*test_percent,)
    Nvalidation = round(len(days)*validation_percent,)
    Ntrain = len(days) - Ntest - Nvalidation
    
    # REPARTO POR DIAS
    train = []
    for i in range(0,Ntrain):
        train.append(i)
    #train = np.random.choice(train, size = len(train), replace = False).tolist()
    validation = []
    for i in range(Ntrain,Ntrain+Nvalidation):
        validation.append(i)
    #validation = np.random.choice(validation, size = len(validation), replace = False).tolist()
    test = []
    for i in range(Ntrain+Nvalidation,Ntrain+Nvalidation+Ntest):
        test.append(i)
    #test = np.random.choice(test, size = len(test), replace = False).tolist()
    
    train_set = []
    validation_set = []
    test_set = []
    for element in train:
        train_set.append(days[element])
    for element in validation:
        validation_set.append(days[element])
    for element in test:
        test_set.append(days[element])
    
    return train_set, test_set, validation_set

def generate_train_validation(days, validation_percent):
    
    Nvalidation = round(len(days)*validation_percent,)
    Ntrain = len(days) - Nvalidation
    
    # REPARTO POR DIAS
    train = []
    for i in range(0,Ntrain):
        train.append(i)
    #train = np.random.choice(train, size = len(train), replace = False).tolist()
    validation = []
    for i in range(Ntrain,Ntrain+Nvalidation):
        validation.append(i)
    #validation = np.random.choice(validation, size = len(validation), replace = False).tolist()

    train_set = []
    validation_set = []
    for element in train:
        train_set.append(days[element])
    for element in validation:
        validation_set.append(days[element])
    
    return train_set, validation_set

def activity_reports(Xday,Yday):
    reports = []
    all_report = generate_empty_activity_dict(array=False)
    day_report = generate_empty_activity_dict(array=True)
    counter = 1
    for day in Yday:
        report_dict = generate_empty_activity_dict(array=False)
        for activity in day:
            if(counter not in day_report[activity]):
                day_report[activity].append(counter)
            report_dict[activity] += 1
            all_report[activity] += 1
        reports.append(report_dict)
        counter += 1
    return reports, all_report, day_report

def daily_stack(Xventana,Yventana):
    Xday = []
    Yday = []
    cosHour_ant = Xventana[0][1] # VALOR INICIAL DE HORA COSENO
    stack_X = []
    stack_Y = []
    for i,ventana in enumerate(Xventana):
        if(ventana[1] >= 0.5 and cosHour_ant < 0.5): # CAMBIO DE DIA
            Xday.append(stack_X)
            Yday.append(stack_Y)
            stack_X = []
            stack_Y = []
        stack_X.append(ventana)
        stack_Y.append(Yventana[i])
        cosHour_ant = ventana[1]
    Xday.append(stack_X)
    Yday.append(stack_Y)
    
    del Xventana, Yventana
    
    # GENERO REPORTES DE ACTIVIDADES
    reports, all_report, day_report = activity_reports(Xday,Yday)
    
    # GENERO EL ARRAY ALEATORIO
    test_days = [4,5,6,9,10,11,14,16,17,27,34,38] # USER 1
    #test_days = [5,13,18,21,23,24,46,47,49,50,59,61] # USER 2

    train_validation_days = []

    for day in range(0,len(Yday)):
    	if(day not in test_days):
    		train_validation_days.append(day)
    Xtest = []
    Ytest = []
    
    for day in test_days:
        for element in Xday[day-1]:
            Xtest.append(element)
        for element in Yday[day-1]:
            Ytest.append(element)
            
    Xtest = np.concatenate(Xtest,axis=0)
    Xtest = np.reshape(Xtest,(-1,tamanoVENTANA*42+2))
    Ytest = np.array(Ytest)   
    
    Xtrain_validation = []
    Ytrain_validation = []
    
    for day in train_validation_days:
        for element in Xday[day-1]:
            Xtrain_validation.append(element)
        for element in Yday[day-1]:
            Ytrain_validation.append(element)
            
    Xtrain_validation = np.concatenate(Xtrain_validation,axis=0)
    Xtrain_validation = np.reshape(Xtrain_validation,(-1,tamanoVENTANA*42+2))
    Ytrain_validation = np.array(Ytrain_validation)  

    #return reports, all_report, day_report
    return Xtrain_validation, Ytrain_validation, Xtest, Ytest

def aleatory_processing_rows(Xventana,Yventana):
    
    train_set = []
    validation_set = []
    
    for num_activity in list(actividades.values()): # SELECCIONO CADA ACTIVIDAD DEL DATASET
        try:
            # OBTENGO LOS INDICES DE CADA VENTANA QUE INTERVIENE EN LA ACTIVIDAD
            v_actividad = np.where(Yventana == num_activity)[0].tolist()
            actividad_ant = v_actividad[0]
            
            # SEPARO POR BLOQUES LAS VENTANAS CONSECUTIVAS DE UNA MISMA ACTIVIDAD
            index_matrix = []
            stack = []                   
            for activity in v_actividad:
                if(activity != actividad_ant+1 and activity != v_actividad[0]): # ACTIVIDAD NO CONSECUTIVA                                               
                    index_matrix.append(stack)
                    stack = []
                stack.append(activity)
                actividad_ant = activity
            index_matrix.append(stack)            
            
            # ALEATORIZO LOS BLOQUES DE ACTIVIDADES CONSECUTIVAS
            aleatorio = np.random.choice(len(index_matrix), size = len(index_matrix), replace = False).tolist()
            index_matrix_aleatory = []
            for activity_block in aleatorio:
                index_matrix_aleatory.append(index_matrix[activity_block])
                
            # METO EN LA MISMA PILA TODAS LAS VENTANAS ORDENADAS
            buffer_activity = []
            for activity_block in index_matrix_aleatory:
                activity_block = np.random.choice(activity_block, size = len(activity_block), replace = False).tolist()
                for window in activity_block:
                    buffer_activity.append(window)
            
            # SEPARO LA PILA EN LAS MEMORIAS CORRESPONDIENTES A TRAINING, TEST Y VALIDACION
            train,validation = generate_train_validation(buffer_activity, validation_percent)
            
            for i in train:
                train_set.append(i)
            if(num_activity != 100):
                for i in validation:
                    validation_set.append(i)
            else:
                for i in validation:
                    train_set.append(i)
        except:
            print("NO EXISTEN ACTIVIDADES DEL TIPO: " + str(get_key(actividades,num_activity)))
    
    train_set = np.random.choice(train_set, size = len(train_set), replace = False).tolist()
    validation_set = np.random.choice(validation_set, size = len(validation_set), replace = False).tolist()
    
    return train_set, validation_set

def delete_rows(Xtrain_validation,Ytrain_validation):
    filtered_rows = []
    for num_activity in list(actividades.values()): # SELECCIONO CADA ACTIVIDAD DEL DATASET
        # OBTENGO LOS INDICES DE CADA VENTANA QUE INTERVIENE EN LA ACTIVIDAD
        v_actividad = np.where(Ytrain_validation == num_activity)[0].tolist()
        actividad_ant = v_actividad[0]
        
        # SEPARO POR BLOQUES LAS VENTANAS CONSECUTIVAS DE UNA MISMA ACTIVIDAD
        index_matrix = []
        stack = []                   
        for activity in v_actividad:
            if(activity != actividad_ant+1 and activity != v_actividad[0]): # ACTIVIDAD NO CONSECUTIVA                                               
                index_matrix.append(stack)
                stack = []
            stack.append(activity)
            actividad_ant = activity
        index_matrix.append(stack)
        
        # PROCESO DE FILTRADO DE ACTIVIDADES
        key = get_key(actividades,num_activity)
        if key in filtros.keys():
            index_matrix_filter = []
            for i in range(0, len(index_matrix)):  
                buffer = []
                if(len(index_matrix[i])>(1/filtros[key])): # SOLO TRUNCO LAS SERIES CON MAS DE 20 REGISTROS
                    filas = round((len(index_matrix[i]))*(filtros[key]/2),0)                        
                    for j in range(0, len(index_matrix[i])):
                        if(j < filas or j >= len(index_matrix[i]) - filas):
                            buffer.append(index_matrix[i][j])                           
                else:
                    for j in range(0, len(index_matrix[i])):
                        buffer.append(index_matrix[i][j])
                index_matrix_filter.append(buffer)
            index_matrix = []
            index_matrix = index_matrix_filter
        for activity_window in index_matrix:
            for window in activity_window:
                filtered_rows.append(window)
    filtered_rows = sorted(filtered_rows)
    Xtrain_validation = Xtrain_validation[filtered_rows]
    Ytrain_validation = Ytrain_validation[filtered_rows]
    
    return Xtrain_validation,Ytrain_validation
            
def balance_data(Xtrain_validation,Ytrain_validation):

    # OBTENGO NUMERO TOTAL DE MUESTRAS POR ACTIVIDAD
    reports = collections.Counter(Ytrain_validation)

    # OBTENGO EL INDICE DE ACTIVIDAD CON MAYOR NUMERO DE MUESTRAS
    max_act = max(reports.items(), key=operator.itemgetter(1))[0]
    max_samples = reports[max_act]

    # ELIMINO DEL DICCIONARIO LA INFORMACION CORRESPONDIENTE A LA ACTIVIDAD CON MAS MUESTRAS
    del reports[max_act]
    del reports[13] # ELIMINO DORMIR YA QUE HAY MUESTRAS SUFICIENTES

    for num_activity in reports.keys():
        # OBTENGO LOS INDICES DE CADA VENTANA QUE INTERVIENE EN LA ACTIVIDAD
        v_actividad = np.where(Ytrain_validation == num_activity)[0].tolist()
        actividad_ant = v_actividad[0]

        # SEPARO POR BLOQUES LAS VENTANAS CONSECUTIVAS DE UNA MISMA ACTIVIDAD
        index_matrix = []
        stack = []                   
        for activity in v_actividad:
            if(activity != actividad_ant+1 and activity != v_actividad[0]): # ACTIVIDAD NO CONSECUTIVA                                               
                index_matrix.append(stack)
                stack = []
            stack.append(activity)
            actividad_ant = activity
        index_matrix.append(stack)      

        report_samples = reports[num_activity]
        # REPITO EL BUCLE HASTA QUE ALCANCE EL TAMAÑO DE LA ACTIVIDAD MAS GRANDE
        num_samples = 0
        while num_samples < max_samples:               
            try:
                block = np.append(block,np.array(index_matrix[randrange(0,len(index_matrix))]))
            except:
                block = np.array(index_matrix[randrange(0,len(index_matrix))])
            num_samples = len(block)
        Xtrain_validation = np.concatenate((Xtrain_validation,Xtrain_validation[block]),axis=0)
        Ytrain_validation = np.concatenate((Ytrain_validation,Ytrain_validation[block]),axis=0)
        del block
    return Xtrain_validation,Ytrain_validation

def share_data(Xtrain_validation, Ytrain_validation):
    
    # CARGO LOS DATOS DEL OTRO USUARIO
    Xshare = np.load(share_path + 'data-xtrainvaltest.npy', allow_pickle=True)
    Yshare = np.load(share_path + 'data-ytrainvaltest.npy', allow_pickle=True)
    
    for activity in share_activities:
        lst_index = np.where(Yshare==actividades[activity])[0]
        Xtrain_validation = np.concatenate((Xtrain_validation,Xshare[lst_index]),axis=0)
        Ytrain_validation = np.concatenate((Ytrain_validation,Yshare[lst_index]),axis=0)
    
    del Xshare, Yshare
    
    return Xtrain_validation,Ytrain_validation

if __name__ == '__main__':
    sensores = []
    sensors_list = []
    print("READ DATA")
    sensores = read_data(sensores,sensors_list)
    print("READ ACTIVITIES")
    activities = read_activities("activity_user_1")
    print("GET STADISTICS")
    dicStatistics = get_statistics(sensores,sensors_list)
    print("LOAD DATASET")
    filas_procesadas = load_dataset(sensores,date_ini,date_end,dicStatistics)
    del sensores,activities
    # ELIMINO LA ACTIVIDAD MEDITATE PORQUE ESTA AGRUPADA CON RELAX
    del actividades["MEDITATE"]
    print("CONVERT PROCCESED ROWS")
    Xventana, Yventana, T = convert_proccesed_rows(filas_procesadas)
    del filas_procesadas
    print("SLIDING WINDOW")
    Xventana, Yventana = sliding_window(Xventana,Yventana,T)
    del T
    print("DAILY STACK")
    #reports, all_report, day_report = daily_stack(Xventana,Yventana)
    Xtrain_validation, Ytrain_validation, Xtest, Ytest = daily_stack(Xventana,Yventana)
    del Xventana,Yventana
    print("DELETE ROWS")
    Xtrain_validation, Ytrain_validation = delete_rows(Xtrain_validation, Ytrain_validation)
    print("SAVING DATA")
    np.save(save_path + 'data-xtrainvaltest.npy', Xtrain_validation)
    np.save(save_path + 'data-ytrainvaltest.npy', Ytrain_validation)
    print("SHARE DATA")
    Xtrain_validation, Ytrain_validation = share_data(Xtrain_validation, Ytrain_validation)
    print("BALANCE DATA")
    Xtrain_validation, Ytrain_validation = balance_data(Xtrain_validation, Ytrain_validation)
    print("ALEATORY PROCESSING ROWS")
    train_set, validation_set = aleatory_processing_rows(Xtrain_validation,Ytrain_validation)
    
    split = dict()
    split = {"train":train_set, "validation": validation_set}
    
    print("SAVING DATA")
    for group in split:
        # TRAIN
        np.save(save_path + 'data-x' + group + '.npy', Xtrain_validation[split[group],:])
        np.save(save_path + 'data-y' + group + '.npy', Ytrain_validation[split[group]])
        
    np.save(save_path + 'data-xtest.npy', Xtest)
    np.save(save_path + 'data-ytest.npy', Ytest)
        
    # ACTIVITIES
    np.save(save_path + 'data-act.npy', actividades)

    
def getData():
    
    # TRAIN
    Xtrain = np.load(save_path + 'data-xtrain.npy', allow_pickle=True)
    Ytrain = np.load(save_path + 'data-ytrain.npy', allow_pickle=True)
    
    # TEST
    Xtest = np.load(save_path + 'data-xtest.npy', allow_pickle=True)
    Ytest = np.load(save_path + 'data-ytest.npy', allow_pickle=True)
    
    # VALIDATION
    Xvalidation = np.load(save_path + 'data-xvalidation.npy', allow_pickle=True)
    Yvalidation = np.load(save_path + 'data-yvalidation.npy', allow_pickle=True)
    
    # ACTIVITIES
    dictAct = np.load(save_path + 'data-act.npy', allow_pickle=True).item()
        
    return Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation, dictAct   
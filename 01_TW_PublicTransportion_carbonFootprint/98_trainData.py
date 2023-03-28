import urllib.request as request
import json
import pandas as pd
import csv
import chardet


src = [
    "https://ods.railway.gov.tw/tra-ods-web/ods/download/dataResource/f0906cb8dcee4dfd9eb5f8a9a2bd0f5a",
    "https://ods.railway.gov.tw/tra-ods-web/ods/download/dataResource/0518b833e8964d53bfea3f7691aea0ee"
       ]

srcCSV = [
    "https://data.taipei/api/dataset/4acb4911-0360-4063-808d-fcee629508b3/resource/893c2f2a-dcfd-407b-b871-394a14105532/download"
]



#============================
#creat TRA station information
df =pd.DataFrame({"lineName":[], "stationCode":[], "stationName":[], "city":[], "staMil":[]}) #各站資料庫
stationFrame = {} #以縣市歸類火車站

for i in src:
    with request.urlopen(i) as response:
        data = json.load(response)
        for station in data:
            if "lineName" in station:
                stationData = pd.DataFrame({"lineName": [station["lineName"]], "stationCode":[int(station["fkSta"])], "staMil":[float(station["staMil"])]})
                df = pd.concat([df, stationData], ignore_index=True)
            elif "stationCode" in station:
                cities = station["stationAddrTw"][:3]
                if cities not in stationFrame:
                   stationFrame[cities] = []
                stationFrame[cities].append(station["stationName"])
                stationData = {"stationCode" : int(station["stationCode"]), "stationName" : station["stationName"], "city": cities}
                df.loc[df["stationCode"] == stationData["stationCode"], "stationName"] = stationData["stationName"]
                df.loc[df["stationCode"] == stationData["stationCode"], "city"] = stationData["city"]


stationList = []
for i in stationFrame:
    stationList += stationFrame[i]
stationFrame = pd.DataFrame.from_dict(stationFrame, orient="index").T
#TRA站到站的里程表
sTos = pd.DataFrame(columns=stationList, index=stationList)

for i in sTos:
    sTos[i][i] = "-"

lines = list(set(i for i in df["lineName"]))

# n=0
# for line in lines:
#     print(line)
#     n += 1
#     print(n)
#     for name in df.loc[df["lineName"] == line, "stationName"]:
#         if str(name) != "nan":
#             origin = df.loc[(df["lineName"] == line)&(df["stationName"] == name), "staMil"]
#             for name1 in df.loc[df["lineName"] == line, "stationName"]:
#                 if str(name1) != "nan":
#                     destination = df.loc[(df["lineName"] == line)&(df["stationName"] == name1), "staMil"]
#                     dist = float(destination) - float(origin)
#                     if name != name1:
#                         sTos[name][name1]=abs(dist)




#=========================
#捷運站，因為轉乘一定換車，所以較好運算
#高雄捷運、台中捷運、桃園捷運
mrt = {
"KRTC":{
    "R_line": 
        {"小港": 0, "高雄國際機場": 1.49, "草衙": 3.32, "前鎮高中(五甲)": 4.46, "凱旋": 5.61, "獅甲(勞工公園)": 6.87, "三多商圈": 7.83, "中央公園": 9.11, "美麗島": 9.85, "高雄車站": 10.73, "後驛(高醫大)": 11.74, "凹子底": 12.7, "巨蛋(三民家商)": 13.69, "生態園區": 14.94, "左營(高鐵)": 16.53, "世運(國家體育園區)": 18.2, "油廠國小(中山大學附中)": 18.98, "楠梓加工區": 20.28, "後勁": 21.29, "都會公園": 22.36, "青埔": 24.05, "橋頭糖廠": 25.14, "橋頭火車站": 26, "南岡山": 28.81},
    "O_line": 
        {"西子灣(中山大學)":0,"鹽埕埔":1.34,"市議會(舊址)":2.63,"美麗島":3.57,"信義國小":4.4,"文化中心":5.05,"五塊厝":6.09,"技擊館":6.83,"衛武營":7.53,"鳳山西站(高雄市議會)":8.28,"鳳山":9.02,"大東":9.87,"鳳山國中":10.91,"大寮(前庄)":13.13}
        },
"TCMRT": {
    "G_line":
        {"北屯總站":0, "舊社":1.01, "松竹":1.69, "四維國小（二分埔）":3.34, "文心崇德":4.2, "文心中清（天津商圈）":5.67, "文華高中":6.73, "文心櫻花":7.55, "(台中)市政府":8.35, "水安宮":9.37, "文心森林公園":10.23, "南屯（文心五權西）":10.77, "豐樂公園":11.65, "大慶（中山醫大）":13.25, "九張犁":14.17, "九德":14.86, "烏日":15.89, "高鐵台中站":16.96}
        },
"TYMC": {
    "Airport_line":
        {"台北車站(AP_line)": 0, "三重(AP_line)": 4.1, "新北產業園區(AP_line)": 7.6, "新莊副都心": 9, "泰山": 9.9, "泰山貴和（明志科大）": 12.7, "體育大學（龜山樂善）": 17.3, "長庚醫院": 20, "林口": 21.2, "緊急停靠站": 25.1, "山鼻": 29.6, "坑口": 31.6, "機場第一航廈": 34.8, "機場第二航廈": 35.8, "機場第三航廈": 36.4, "機場旅館": 37.1, "大園": 39.1, "橫山": 41.4, "領航（大園國際高中）": 42.9, "高鐵桃園站": 44.5, "桃園體育園區": 46.3, "興南": 49.2, "環北": 50.8, "老街溪": 51.33}
        }
}
transStation = {
    "KRTC": {"R_line": "美麗島", "O_line": "美麗島"},
    "TCMRT": {"G_line": ""},
    "TYMC": {"Airport_line": ""}
}
#製作dataframe(高捷、中捷、桃捷)
mrtList0 = {"system":[], "line":[], "station":[], "mileage":[]}
for i in mrt:
    for j in mrt[i]:
        for k in mrt[i][j]:
            mrtList0["system"].append(i)
            mrtList0["line"].append(j)
            mrtList0["station"].append(k)
            mrtList0["mileage"].append(mrt[i][j][k])
mrtList =pd.DataFrame(mrtList0)

#處理台北捷運站名跟里程表(csv格式)
TRTC_CSV = "95_TRTC.csv"
TRTC_stations = []
TRTC_odometer = []

with open(TRTC_CSV, "rb") as response:
    encoding = chardet.detect(response.read())["encoding"]

with open(TRTC_CSV, mode= "r", encoding= encoding) as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] not in TRTC_stations:
            TRTC_stations.append(row[0]) #提取站名
        TRTC_odometer.append(row)
    for i in TRTC_odometer:
        i.append("TRTC")
    del TRTC_stations[0]


#creat odometer

odometer_index = [i for i in mrtList["station"]] + TRTC_stations
odometer_columns= [i for i in mrtList["station"]] + TRTC_stations

odometer = pd.DataFrame(index= odometer_index, columns= odometer_columns)


for origin in mrtList["station"]:
    for destination in mrtList["station"]:
        origin_system = mrtList.loc[(mrtList["station"] == origin), "system"].astype(str).tolist()[0]
        dest_system = mrtList.loc[(mrtList["station"] == destination), "system"].astype(str).tolist()[0]
        origin_line = mrtList.loc[(mrtList["station"] == origin), "line"].astype(str).tolist()[0]
        dest_line = mrtList.loc[(mrtList["station"] == destination), "line"].astype(str).tolist()[0]
        origin_mileage = mrtList.loc[(mrtList["station"] == origin), "mileage"].astype(float).tolist()[0]
        dest_mileage = mrtList.loc[(mrtList["station"] == destination), "mileage"].astype(float).tolist()[0]
        originTranStan = transStation[origin_system][origin_line]
        destTranStan = transStation[dest_system][dest_line]
        origin_mileage_T = mrtList.loc[(mrtList["station"] == originTranStan) & (mrtList["line"] == origin_line), "mileage"]
        dest_mileage_T = mrtList.loc[(mrtList["station"] == destTranStan) & (mrtList["line"] == dest_line), "mileage"]
        if (origin == destination):
            odometer.loc[origin, destination] = "-"
        else:
            if (origin_system == dest_system) and (origin_line == dest_line):
                odometer.loc[origin,destination] = abs(float(dest_mileage)-float(origin_mileage))
            elif (origin_system == dest_system) and (origin_line != dest_line):
                oMileage = abs(float(origin_mileage_T)-float(origin_mileage))
                dMileage = abs(float(dest_mileage_T)-float(dest_mileage))
                odometer.loc[origin, destination] = oMileage + dMileage

#focuse on analyst TRTC
for item in TRTC_odometer:
    if (item[0] in odometer_index) and (item[0] != item[1]):
        odometer.loc[item[0], item[1]] = float(item[5])
    elif (item[0] in odometer_index) and (item[0] == item[1]):
        odometer.loc[item[0], item[1]] = "-"

#TYMR and TRTC
TRTC_transport_station = {"台北車站": "台北車站(AP_line)" , "三重": "三重(AP_line)", "新北產業園區": "新北產業園區(AP_line)", "丹鳳": "泰山貴和（明志科大）"}

totDis = []

for TRTC_data in TRTC_odometer:
    if TRTC_data[1] in [j for j in TRTC_transport_station]:
        TYMC_init = TRTC_transport_station[TRTC_data[1]]
        for TYMC_dest in mrt["TYMC"]["Airport_line"]:
            TYMC_dis = odometer.loc[TYMC_init, TYMC_dest]
            if TYMC_dis == "-":
                TYMC_dis = 0
            dis = float(TRTC_data[5]) + TYMC_dis
            if [TRTC_data[0], TYMC_dest] not in [i[0:2] for i in totDis]:
                a = [TRTC_data[0], TYMC_dest, dis, TRTC_data[1], TYMC_init]
                totDis.append(a)
            for data in totDis:
                if ([TRTC_data[0], TYMC_dest] == data[0:2]) and (data[2]>dis):
                    totDis.remove(data)
                    totDis.append([TRTC_data[0], TYMC_dest, dis, TRTC_data[1], TYMC_init])

#強制讓TRTC_transport_station中的同名站距離變為0
# for station in TRTC_transport_station: 
#     for data in totDis:
#         if ([station, TRTC_transport_station[station]] == data[0:2]):
#             totDis.remove(data)
#             totDis.append([station, TRTC_transport_station[station], 0, station, TRTC_transport_station[station]])

#確定上面沒有發生錯誤                
for i in totDis:
    if i[0:2] in [["台北車站", "台北車站(AP_line)"] , ["三重", "三重(AP_line)"], ["新北產業園區", "新北產業園區(AP_line)"], ["丹鳳", "泰山貴和（明志科大）"]] and i[2] != 0:
        print("AP line到北捷,相同站的距離有問題")

#測試toDis中是否有重複的起迄站
a = [i[0:2] for i in totDis]
for i in a:
    c = a.count(i)
    if c > 1:
        print("totDis has dulpicate station")

#save totDis to the odometer
for station in totDis:
    odometer.loc[station[0], station[1]] = station[2]
    odometer.loc[station[1], station[0]] = station[2]

 

#===================
#save to excel
excel = "02_train.xlsx"
write = pd.ExcelWriter(excel)
df.to_excel(write, sheet_name= "TRAstationData")
sTos.to_excel(write, sheet_name="TRAstation2station")
stationFrame.to_excel(write, sheet_name = "TRAcityStation")
odometer.to_excel(write, sheet_name= "MRT_odometer")
write.close()

#stationList = list(set(stationName for stationName in df["stationName"] if stationName != ""))

#wiki TRTC station name check
# lineStations = [
#     "南港展覽館", "南港軟體園區", "東湖", "葫洲", "大湖公園", "內湖", "文德（碧湖公園）", "港墘", "西湖（德明科大）", "劍南路（美麗華）", "大直（實踐大學）", "松山機場", "中山國中", "南京復興", "忠孝復興", "大安", "科技大樓", "六張犁", "麟光", "辛亥", "萬芳醫院", "萬芳社區", "木柵", "動物園", "淡水", "紅樹林", "竹圍", "關渡", "忠義", "復興崗", "新北投", "北投", "奇岩", "唭哩岸", "石牌（榮總）", "明德", "芝山", "士林", "劍潭（北藝中心）", "圓山", "民權西路", "雙連（馬偕紀念醫院）", "中山", "台北車站", "台大醫院", "中正紀念堂（南門）", "東門", "大安森林公園", "大安", "信義安和", "台北101/世貿", "象山", "廣慈/奉天宮", "松山", "南京三民", "台北小巨蛋", "南京復興", "松江南京", "中山", "北門（大稻埕南）", "西門", "小南門", "中正紀念堂（南門）", "古亭", "台電大樓", "公館（台灣大學）", "萬隆", "景美", "小碧潭（新店高中）", "大坪林", "七張", "新店區公所", "新店（碧潭）", "蘆洲（蘆洲李宅）", "三民高中（空中大學）", "徐匯中學", "三和國中", "三重國小", "迴龍（樂生）", "丹鳳", "輔大", "新莊（新莊廟街）", "頭前庄（臺北醫院）", "先嗇宮", "三重", "菜寮（新北市立醫院）", "台北橋", "大橋頭（大橋國小）", "民權西路", "中山國小（晴光商圈）", "行天宮", "松江南京", "忠孝新生（台北科大）", "東門", "古亭", "頂溪", "永安市場", "景安", "南勢角", "南港展覽館", "南港", "昆陽", "後山埤（五分埔商圈）", "永春", "市政府", "國父紀念館", "忠孝敦化", "忠孝復興", "忠孝新生（台北科大）", "善導寺（華山）", "台北車站", "西門", "龍山寺（艋舺商圈）", "江子翠", "新埔", "板橋", "府中（林家花園）", "亞東醫院", "海山", "土城", "永寧", "頂埔", "大坪林", "十四張（莊敬高職）", "秀朗橋", "景平", "景安", "中和", "橋和", "中原", "板新", "板橋", "新埔民生", "頭前庄（臺北醫院）", "幸福", "新北產業園區"
#     ]

# lineStations = list(set(lineStations))

# print("excel version")
# for station in lineStations:
#     if station not in TRTC_stations:
#         print(station)
        
# print("python version")
# for station in TRTC_stations:
#     if station not in lineStations:
#         print(station)
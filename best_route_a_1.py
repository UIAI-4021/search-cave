# -*- coding: utf-8 -*-
"""Best route A*.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tIWGGVIJur5PUyFiH7TsEpCW-hUQ2Tp-
"""

import heapq
import time
import pandas as pd
df = pd.read_csv('Flight_Data.csv')
Start = time.time()
def astar(graph, graphP, graphF, graphA, start, end, Latitude, Longitude, Altitude):
    open_set = [(0, start)]
    closed_set = set()
    previous = {node: None for node in graph}
    previousP = {node: None for node in graph}
    previousD = {node: None for node in graph}
    previousF = {node: None for node in graph}
    previousA = {node: None for node in graph}
    cost_so_far = {start: 0}
    while open_set:
        current_cost, current_node = heapq.heappop(open_set)
        if current_node == end:
            path = []
            pathP = []
            pathD = []
            pathF = []
            pathA = []
            current_node = end
            while current_node != start:
                path.append(current_node)
                pathP.append(previousP[current_node])
                pathD.append(previousD[current_node])
                pathF.append(previousF[current_node])
                pathA.append(previousA[current_node])
                current_node = previous[current_node]
            path.append(start)
            Path = {}
            Path["Node"] = path[::-1]
            Path["Price"] = pathP[::-1]
            Path["Distance"] = pathD[::-1]
            Path["FlyTime"] = pathF[::-1]
            Path["Airline"] = pathA[::-1]
            Path["end"] = cost_so_far[end]
            return Path
        closed_set.add(current_node)
        for neighbor, cost in graph[current_node].items():
            if neighbor in closed_set:
                continue
            tentative_cost = cost_so_far[current_node] + cost
            heuristic = Distance0(Latitude, Longitude, Altitude, neighbor, end)
            heapq.heappush(open_set, (tentative_cost + heuristic, neighbor))
            previous[neighbor] = current_node
            previousP[neighbor] = graphP[current_node][neighbor]
            previousD[neighbor] = graph[current_node][neighbor]
            previousF[neighbor] = graphF[current_node][neighbor]
            previousA[neighbor] = graphA[current_node][neighbor]
            cost_so_far[neighbor] = tentative_cost
    return None
def Distance0(Latitude, Longitude, Altitude, neighbor, end):
    return (min(abs(Latitude[neighbor] - Latitude[end]), 360 - abs(Latitude[neighbor] - Latitude[end])) ** 2 + min(abs(Longitude[neighbor] - Longitude[end]), 360 - abs(Longitude[neighbor] - Longitude[end])) ** 2) ** 0.5

graph = {}
graphD = {}
graphF = {}
graphA = {}
Country = {}
City = {}
Latitude = {}
Longitude = {}
Altitude = {}
for _, row in df.iterrows():
    weight = row.Price
    Distance = row.Distance
    FlyTime = row.FlyTime
    Airline = row.Airline
    Country[row.DestinationAirport] = row.DestinationAirport_Country
    City[row.DestinationAirport] = row.DestinationAirport_City
    Latitude[row.DestinationAirport] = row.DestinationAirport_Latitude
    Longitude[row.DestinationAirport] = row.DestinationAirport_Longitude
    Altitude[row.DestinationAirport] = row.DestinationAirport_Altitude
    graph.setdefault(row.SourceAirport, {})
    graphD.setdefault(row.SourceAirport, {})
    graphF.setdefault(row.SourceAirport, {})
    graphA.setdefault(row.SourceAirport, {})
    graph.setdefault(row.DestinationAirport, {})
    graphD.setdefault(row.DestinationAirport, {})
    graphF.setdefault(row.DestinationAirport, {})
    graphA.setdefault(row.DestinationAirport, {})
    graph[row.SourceAirport][row.DestinationAirport] = weight
    graphD[row.SourceAirport][row.DestinationAirport] = Distance
    graphF[row.SourceAirport][row.DestinationAirport] = FlyTime
    graphA[row.SourceAirport][row.DestinationAirport]= Airline
Airport = input()
Airport = Airport.split(' - ')
Path = astar(graphD, graph, graphF, graphA, Airport[0], Airport[1], Latitude, Longitude, Altitude)
s = f"""A* Algorithm
Execution Time: {time.time() - Start}
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
"""
TP = sum(Path["Price"])
TD = sum(Path["Distance"])
TT = sum(Path["FlyTime"])
n = len(Path["Airline"])

for i in range(1, n + 1):
    s += f"""Flight #{i} ({Path["Airline"][i - 1]})
From: {Path["Node"][i - 1]} - {City[Path["Node"][i - 1]]}, {Country[Path["Node"][i - 1]]}
To: {Path["Node"][i]} - {Country[Path["Node"][i]]}, {City[Path["Node"][i]]}
Duration: {Path["Distance"][i - 1]}km
Time: {Path["FlyTime"][i - 1]}h
Price: {Path["Price"][i - 1]}$
----------------------------
"""
s += f"""Total Price: {TP}
Total Duration: {TD}
Total Time: {TT}
"""
with open('14-UIAI4021-PR1-Q1(A*).txt', 'w') as file:
  file.write(s)
import pandas as pd
import xml.etree.ElementTree as et


def get_average_travel_time():
    xtree = et.parse(r"C:/Users/kriti/OneDrive/Desktop/capstone/amman.tripinfo.xml")
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        travel_time = node.attrib.get("duration")
        rows.append({"travel_time": travel_time})

    columns = ["travel_time"]
    travel_time = pd.DataFrame(rows, columns=columns).astype("float64")
    return travel_time["travel_time"].mean()


def get_average_waiting_time_test_AI():
    xtree = et.parse(r"C:/Users/kriti/OneDrive/Desktop/capstone/amman.tripinfo.xml")
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        travel_time = node.attrib.get("waitingTime")
        rows.append({"waitingTime": travel_time})

    columns = ["waitingTime"]
    waitingTime = pd.DataFrame(rows, columns=columns).astype("float64")
    return waitingTime["waitingTime"].mean()


def get_average_waiting_time_test():
    xtree = et.parse(r"C:/Users/kriti/OneDrive/Desktop/capstone/amman.tripinfo.xml")
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        travel_time = node.attrib.get("waitingTime")
        rows.append({"waitingTime": travel_time})

    columns = ["waitingTime"]
    waitingTime = pd.DataFrame(rows, columns=columns).astype("float64")
    return waitingTime["waitingTime"].mean()


def get_average_length():
    xtree = et.parse(r"C:/Users/kriti/OneDrive/Desktop/capstone/amman.tripinfo.xml")
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        routeLength = node.attrib.get("routeLength")
        rows.append({"routeLength": routeLength})

    columns = ["routeLength"]
    routeLength = pd.DataFrame(rows, columns=columns).astype("float64")
    return routeLength["routeLength"].mean() * 0.001


def get_average_CO2():
    xtree = et.parse(r"C:/Users/kriti/OneDrive/Desktop/capstone/amman.tripinfo.xml")
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        for child in node.iter():
            co2_emission = child.attrib.get("CO2_abs")
            rows.append({"co2_emission": co2_emission})

    columns = ["co2_emission"]
    co2_emission = pd.DataFrame(rows, columns=columns).astype("float64")
    return co2_emission["co2_emission"].mean() * 0.0001


def get_average_fuel():
    xtree = et.parse(r"C:/Users/kriti/OneDrive/Desktop/capstone/amman.tripinfo.xml")
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        for child in node.iter():
            fuel_cons = child.attrib.get("fuel_abs")
            rows.append({"fuel_cons": fuel_cons})

    columns = ["fuel_cons"]
    fuel_cons = pd.DataFrame(rows, columns=columns).astype("float64")
    return fuel_cons["fuel_cons"].mean() * 0.00001


def get_total_cars():
    xtree = et.parse(r"C:/Users/kriti/OneDrive/Desktop/capstone/amman.tripinfo.xml")
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        car_id = node.attrib.get("id")
        rows.append({"id": car_id})

    columns = ["id"]
    id_df = pd.DataFrame(rows, columns=columns)
    return id_df.shape[0]

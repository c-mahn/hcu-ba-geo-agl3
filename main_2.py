# Import von Bibliotheken
# -----------------------------------------------------------------------------

import pickle
import lib_kalman
import matplotlib.pyplot as plt
import numpy as np
import math as m
import copy
import random as r

# Funktionen
# -----------------------------------------------------------------------------


def plot_werte(datenreihen, name=["Messwerte"]):
    """
    Diese Funktion nimmt Datenreihen und plottet diese in ein Diagramm.
    """
    for i, datenreihe in enumerate(datenreihen):
        zeit = range(len(datenreihe))
        if(i == 0):
            plt.plot(zeit, datenreihe, "o")
        else:
            plt.plot(zeit, datenreihe)
    plt.legend(name)
    plt.grid()
    plt.xlabel("")
    plt.ylabel("")
    plt.title(name[0])
    plt.show()


def plot_xy(datenreihen, name=["Messwerte"]):
    """
    Diese Funktion nimmt je zwei Datenreihen und plottet diese in Abhängigkeit
    zueinander in ein Diagramm.
    """
    for i, datenreihe in enumerate(datenreihen):
        if(i == 0):
            plt.plot(datenreihe[0], datenreihe[1], "o")
        else:
            plt.plot(datenreihe[0], datenreihe[1])
    plt.legend(name)
    plt.grid()
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.title(name[0])
    plt.show()


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Klassen
# -----------------------------------------------------------------------------

# Beginn des Programms
# -----------------------------------------------------------------------------

if(__name__ == "__main__"):

    # Umgebungsvariablen
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    verbose = False  # "True" zeigt mehr Debug-Informationen
    filename = "Daten/daten_gruppe1"  # Datei mit Beobachtungen
    data_rate = 1/200  # Festlegen der Datenrate

    # Import von Daten
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    data = load_obj(filename)
    data_position = [list(data["gps"][0]), list(data["gps"][1])]
    data_geschwindigkeit = list(data["v_odo"][0])
    data_ori_aenderung = list(data["dalphaZ_xsens"][0])
    del data

    # Plot von den einzelnen Messreihen
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if(verbose):
        plot_xy([[data_position[0], data_position[1]]],
                ["Position"])
        plot_werte([data_position[0]],
                   ["X"])
        plot_werte([data_position[1]],
                   ["Y"])
        plot_werte([data_geschwindigkeit],
                   ["Geschwindigkeit"])
        plot_werte([data_ori_aenderung],
                   ["Orientierungsänderung"])

    # Messbereiche für Varianzberechnung
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    sigma_x = np.var(data_position[0][8400:8800])
    sigma_y = np.var(data_position[1][8400:8800])
    sigma_geschw = np.var(data_geschwindigkeit[8400:8800])
    sigma_ori_aender = np.var(data_ori_aenderung[8400:8800])
    if(verbose):
        plot_werte([data_position[0][8400:8800]],
                   ["Varianz_X"])
        plot_werte([data_position[1][8400:8800]],
                   ["Varianz_Y"])
        plot_werte([data_geschwindigkeit[8400:8800]],
                   ["Varianz_Geschwindigkeit"])
        plot_werte([data_ori_aenderung[8400:8800]],
                   ["Varianz_Orientierungsänderung"])

    # Erstellen des KalmanFilter-Objektes
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    kalman = lib_kalman.DynKalmanFilterKreis()
    kalman.set_data_rate(data_rate)

    # Berechnung des ersten Ergebnisses
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Startwerte für X0-Vektor
    ori0 = m.atan2(data_position[0][199]-data_position[0][0],
                   data_position[1][199]-data_position[1][0])
    x0_dach_vektor = np.array([[data_position[0][0]],
                               [data_position[1][0]],
                               [ori0],
                               [data_geschwindigkeit[0]],
                               [data_ori_aenderung[0]]])
    del ori0

    # Startwerte für Kovarianzmatrix von X0
    kov_x0_matrix = np.array([[sigma_x],
                              [sigma_y],
                              [0.08**2],  # @Christopher: eventuell anpassen
                              [sigma_geschw],
                              [sigma_ori_aender]])
    kov_x0_matrix = np.diagflat(kov_x0_matrix)

    # Festlegen der Kovarianzmatrix der Beobachtungen
    kov_l_matrix = np.array([[kov_x0_matrix[0][0]],
                             [kov_x0_matrix[1][1]],
                             [kov_x0_matrix[3][3]],
                             [kov_x0_matrix[4][4]]])
    kov_l_matrix = np.diagflat(kov_l_matrix)

    # Festlegen des Störgrößenvektors w
    w_vektor = np.array([[float(x0_dach_vektor[3])],
                         [float(x0_dach_vektor[4])]])

    # Festlegen der Störgrößen-Varianzmatrix
    kov_w_matrix = np.array([[kov_x0_matrix[3][3]],
                             [kov_x0_matrix[4][4]]])
    kov_w_matrix = np.diagflat(kov_w_matrix)

    # Übergeben der Startwerte zum Iterieren
    kalman.set_start_values(x0_dach_vektor,
                            kov_x0_matrix,
                            w_vektor,
                            kov_w_matrix)

    # Erstellung einer Kopie für Aufgabe 3 (Sensorausfall-Simulation)
    kalman_ausfall = copy.copy(kalman)

    # Umwandeln von Messreihen in Beobachtungsvektoren
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    l_vektoren = []
    for i, e in enumerate(data_geschwindigkeit):
        l_vektoren.append(np.array([[data_position[0][i]],
                                    [data_position[1][i]],
                                    [e],
                                    [data_ori_aenderung[i]]]))
    del i, e
    l_vektoren = l_vektoren[1:]

    # Iteratives aktualisieren des Ergebnisses mit neuen Beobachtungen
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    kalman_ergebnisreihen = [[], [], [], [], []]
    print("Running iterative Kalman-Filter (Original)")
    for i, l_vektor in enumerate(l_vektoren):
        """
        Dieser Teil des Kalman-Filters wird iteriert, um schrittweise die
        Prädiktion mit neuen Beobachtungen "zu füttern".
        """
        print(f"{100*i/len(l_vektoren):5.1f}% ", end="")
        print(f"[{int(25*i/len(l_vektoren))*'#'}", end="")
        print(f"{int(25-(25*i/len(l_vektoren)))*' '}] ", end="")
        print(f"Iter.: {i+1}/{len(l_vektoren)}",
              end="\r")
        kalman.praediktion()
        kalman.innovation(l_vektor, kov_l_matrix)
        kalman.gain_matrix()
        ergebnis = kalman.update()[0]
        for j, messwert in enumerate(ergebnis):
            kalman_ergebnisreihen[j].append(messwert)
    print(f"Done!  [{24*'#'}] Iter.: {len(l_vektoren)}/{len(l_vektoren)}")
    del i, l_vektor, j, messwert

# #############################################################################

    # Iteratives aktuallisieren des Ergebnisses mit neuen Beobachtungen (mit
    # Sensorausfall)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Grobe Fehler in den GPS-Beobachtungen (Ausreißer)
    # for i, e in enumerate(l_vektoren):
    #     if(i % 100 == 0):
    #         l_vektoren[i][0] += np.random.normal(0, 1)  # GPS X
    #         l_vektoren[i][1] += np.random.normal(0, 1)  # GPS Y

    # Grobe Fehler über einen Zeitraum
    for i, e in enumerate(l_vektoren):
        if(9000 < i < 10000):
            l_vektoren[i][0] += np.random.normal(0, 1)  # GPS X
            l_vektoren[i][1] += np.random.normal(0, 1)  # GPS Y

    # Grobe Fehler über einen Zeitraum
    # for i, e in enumerate(l_vektoren):
    #     if(9000 < i < 10000):
    #         l_vektoren[i][0] += np.random.normal(0, 1)  # GPS X
    #         l_vektoren[i][1] += np.random.normal(0, 1)  # GPS Y
    #         l_vektoren[i][3] += np.random.normal(0, 0.05)  # IMU

    # Sensorausfall einstellen (Zeiträume mit Komplettausfall)
    # for i, e in enumerate(l_vektoren):
    #     if(9000 < i < 10000):
    #         l_vektoren[i][0] = np.nan  # GPS X
    #         l_vektoren[i][1] = np.nan  # GPS Y
    #         l_vektoren[i][2] = np.nan  # Odometer
    #         l_vektoren[i][3] = np.nan  # IMU

    # Sensorausfall einstellen (GPS nur alle 100)
    # for i, e in enumerate(l_vektoren):
    #     if(i % 100 != 0):
    #         l_vektoren[i][0] = np.nan  # GPS X
    #         l_vektoren[i][1] = np.nan  # GPS Y
    #     elif(r.randint(0, 20) == 0):
    #         l_vektoren[i][0] += np.random.normal(0, 1)  # GPS X
    #         l_vektoren[i][1] += np.random.normal(0, 1)  # GPS Y

    # Erzeugen von Messreihen mit Ausfällen zum Plotten
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    data_position_ausfall = [[], []]
    data_geschwindigkeit_ausfall = []
    data_ori_aenderung_ausfall = []
    for i in l_vektoren:
        if(np.isnan(i[0]) or np.isnan(i[1])):
            data_position_ausfall[0].append(None)
            data_position_ausfall[1].append(None)
        else:
            data_position_ausfall[0].append(float(i[0]))
            data_position_ausfall[1].append(float(i[1]))
        if(np.isnan(i[2])):
            data_geschwindigkeit_ausfall.append(None)
        else:
            data_geschwindigkeit_ausfall.append(float(i[2]))
        if(np.isnan(i[3])):
            data_ori_aenderung_ausfall.append(None)
        else:
            data_ori_aenderung_ausfall.append(float(i[3]))

    # Durchführen des Kalman-Filters
    kalman_ergebnisreihen_ausfall = [[], [], [], [], []]
    print("Running iterative Kalman-Filter (Mit Messfehlern)")
    for i, l_vektor in enumerate(l_vektoren):
        """
        Dieser Teil des Kalman-Filters wird iteriert, um schrittweise die
        Prädiktion mit neuen Beobachtungen "zu füttern".
        """
        print(f"{100*i/len(l_vektoren):5.1f}% ", end="")
        print(f"[{int(25*i/len(l_vektoren))*'#'}", end="")
        print(f"{int(25-(25*i/len(l_vektoren)))*' '}] ", end="")
        print(f"Iter.: {i+1}/{len(l_vektoren)}",
              end="\r")
        kalman_ausfall.praediktion()
        kalman_ausfall.innovation(l_vektor, kov_l_matrix)
        kalman_ausfall.gain_matrix()
        ergebnis = kalman_ausfall.update()[0]
        for j, messwert in enumerate(ergebnis):
            kalman_ergebnisreihen_ausfall[j].append(messwert)
    print(f"Done!  [{24*'#'}] Iter.: {len(l_vektoren)}/{len(l_vektoren)}")
    del i, l_vektor, j, messwert

    # Ausgabe der Ergebnisse als Diagramme
    plot_xy([[data_position_ausfall[0], data_position_ausfall[1]],
             [kalman_ergebnisreihen[0], kalman_ergebnisreihen[1]],
             [kalman_ergebnisreihen_ausfall[0],
              kalman_ergebnisreihen_ausfall[1]]],
            ["Position",
             "Gefiltert",
             "Mit Messfehlern"])
    plot_werte([data_position_ausfall[0],
                kalman_ergebnisreihen[0],
                kalman_ergebnisreihen_ausfall[0]],
               ["X",
                "Gefiltert",
                "Mit Messfehlern"])
    plot_werte([data_position_ausfall[1],
                kalman_ergebnisreihen[1],
                kalman_ergebnisreihen_ausfall[1]],
               ["Y",
                "Gefiltert",
                "Mit Messfehlern"])
    plot_werte([data_geschwindigkeit_ausfall,
                kalman_ergebnisreihen[3],
                kalman_ergebnisreihen_ausfall[3]],
               ["Geschwindigkeit",
                "Gefiltert",
                "Mit Messfehlern"])
    plot_werte([data_ori_aenderung_ausfall,
                kalman_ergebnisreihen[4],
                kalman_ergebnisreihen_ausfall[4]],
               ["Orientierungsänderung",
                "Gefiltert",
                "Mit Messfehlern"])

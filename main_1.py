t# Import von Bibliotheken
# -----------------------------------------------------------------------------

import pickle
import lib_kalman
import matplotlib.pyplot as plt
import numpy as np
import math as m

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
    print("Running iterative Kalman-Filter")
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

    plot_xy([[data_position[0], data_position[1]],
             [kalman_ergebnisreihen[0], kalman_ergebnisreihen[1]]],
            ["GPS-Positionsdaten", "Kalman gefilterte Strecke"])
    plot_werte([data_position[0], kalman_ergebnisreihen[0]],
               ["X", "Gefiltert"])
    plot_werte([data_position[1], kalman_ergebnisreihen[1]],
               ["Y", "Gefiltert"])
    plot_werte([data_geschwindigkeit, kalman_ergebnisreihen[3]],
               ["Geschwindigkeit", "Gefiltert"])
    plot_werte([data_ori_aenderung, kalman_ergebnisreihen[4]],
               ["Orientierungsänderung", "Gefiltert"])

    # Export von Daten
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # with open("out/export.txt") as file:
    #     file.write("Temp")

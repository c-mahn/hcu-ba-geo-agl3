# lib_kalman
# #############################################################################

# lib_kalman beinhaltet die zur Berechnung und Anwendung des dynamischen Kalman
# -Filters benötigten Formeln und Rechenschritte. Es müssen lediglich die
# entsprechenden Daten übergeben werden. Für weitere Hilfe lesen Sie bitte die
# Hilfe-Erläuterungen der einzelnen Funktionen.

# Autoren:
# Maybrit Gießler
# Christopher Mahn

# #############################################################################

# Import von Bibliotheken
# -----------------------------------------------------------------------------

# import functions
import math as m
import numpy as np

# Funktionen
# -----------------------------------------------------------------------------

# Klassen
# -----------------------------------------------------------------------------


class DynKalmanFilterKreis():
    """
    Diese Klasse wird zur Anwendung des dynamischen Kalman-Filters benötigt.
    Hierbei wird von einer kreisförmigen Bewegungsbahn ausgegangen.
    """
    def __init(self):
        pass

    def set_data_rate(self, data_rate):
        self.__delta_t = data_rate

    def set_start_values(self, x_vektor, kov_x_matrix, w_vektor, kov_w_matrix):
        self.__x_dach_vektor = x_vektor
        self.__kov_x_dach_matrix = kov_x_matrix
        self.__split_x_vektor()
        self.__w_vektor = w_vektor
        self.__kov_w_matrix = kov_w_matrix

    def __split_x_vektor(self):
        self.__y = self.__x_dach_vektor[0, 0]
        self.__x = self.__x_dach_vektor[1, 0]
        self.__a = self.__x_dach_vektor[2, 0]
        self.__v = self.__x_dach_vektor[3, 0]
        self.__delta_a = self.__x_dach_vektor[4, 0]

    def praediktion(self):
        """
        Bei der Prädiktion werden die aktuellen Ergebnisse mithilfe der
        Funktionen der Kreisbewegung in die Zukunft prädiziert.
        Die Funktionen werden ohne Linearisierung angewendet!

        Für die neue Transitions- und Störgrößenmatrix werden die Funktionen
        jeweils linearisiert angewendet.

        Außerdem wird die Kovarianzmatrix der Prädiktion berechnet.
        """
        # Aufstellen der Transitionsmatrix
        A = self.__v*self.__delta_t*(m.sin(self.__a+self.__delta_a)-m.sin(self.__a))/self.__delta_a
        B = self.__v*self.__delta_t*(m.cos(self.__a+self.__delta_a)-m.cos(self.__a))/self.__delta_a
        C = -self.__delta_t*(m.cos(self.__a+self.__delta_a)-m.cos(self.__a))/self.__delta_a
        D = self.__delta_t*(m.sin(self.__a+self.__delta_a)-m.sin(self.__a))/self.__delta_a
        E = self.__v*self.__delta_t*(m.sin(self.__a+self.__delta_a))/self.__delta_a+self.__v*self.__delta_t*(m.cos(self.__a+self.__delta_a)-m.cos(self.__a))/(self.__delta_a**2)
        F = self.__v*self.__delta_t*(m.cos(self.__a+self.__delta_a))/self.__delta_a-self.__v*self.__delta_t*(m.sin(self.__a+self.__delta_a)-m.sin(self.__a))/(self.__delta_a**2)
        self.__t_matrix = np.array([[1, 0, A, C, E],
                                    [0, 1, B, D, F],
                                    [0, 0, 1, 0, 1],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1]])

        # Aufstellen der Störgrößenmatrix
        self.__c_matrix = self.__t_matrix[0:, 3:]

        # Prädiktion des nächsten x-Vektors
        y = self.__y-self.__v*self.__delta_t*(m.cos(self.__a+self.__delta_a)-m.cos(self.__a))/self.__delta_a
        x = self.__x+self.__v*self.__delta_t*(m.sin(self.__a+self.__delta_a)-m.sin(self.__a))/self.__delta_a
        a = self.__a+self.__delta_a
        v = self.__v
        delta_a = self.__delta_a
        self.__x_strich_vektor = np.array([[y], [x], [a], [v], [delta_a]])
        del y, x, a, v

        # Berechnung der Kovarianzmatrix der Prädiktion
        self.__kov_x_strich_matrix = self.__t_matrix@self.__kov_x_dach_matrix@np.transpose(self.__t_matrix)+self.__c_matrix@self.__kov_w_matrix@np.transpose(self.__c_matrix)

        # Ausgabe der Berechnungen für eventuelle weitere externe Berechnungen
        return(self.__t_matrix, self.__c_matrix, self.__x_strich_vektor,
               self.__kov_x_strich_matrix)

    def innovation(self, l_vektor_neu, kov_l_matrix_neu):
        """
        Zur Berechnung der Innovation werden die prädizierten Werte und die
        neuen Beobachtungen benötigt, um daraus die Differenz aus Vorhersage
        und Realität (neue Beobachtungen) berechnen zu können.

        Bei der Innovation wird zunächst die neue Designmatrix berechnet und
        dann sowohl die Innovation selbst, als auch die Kovarianzmatrix der
        Innovation berechnet.
        """
        # Aufstellen der Designmatrix
        self.__a_matrix = np.array([[1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1]])

        # Bei Ausfall die Designmatrix und den Beobachtungsvektor verändern
        ausfall_indices = []
        for i, e in enumerate(l_vektor_neu):
            if(np.isnan(e)):
                ausfall_indices.append(i)
        self.__a_matrix = np.delete(self.__a_matrix, ausfall_indices, axis=0)
        l_vektor_neu = np.delete(l_vektor_neu, ausfall_indices, axis=0)
        kov_l_matrix_neu = np.delete(kov_l_matrix_neu, ausfall_indices, axis=1)
        kov_l_matrix_neu = np.delete(kov_l_matrix_neu, ausfall_indices, axis=0)

        # Berechnung der Innovation
        self.__d_vektor = l_vektor_neu-self.__a_matrix@self.__x_strich_vektor

        # Berechnung der Kovarianzmatrix der Innovation
        self.__kov_d_matrix = kov_l_matrix_neu+self.__a_matrix@self.__kov_x_strich_matrix@np.transpose(self.__a_matrix)

        # Ausgabe der Berechnungen für eventuelle weitere externe Berechnungen
        return(self.__a_matrix, self.__d_vektor, self.__kov_d_matrix)

    def gain_matrix(self):
        """
        Die Gain-Matrix stellt die relative Gewichtung der Genauigkeit der
        Vorhersage (Prädiktion) und der jeweiligen Beobachtungen dar.

        Zur Berechnung wird sowohl die Kovarianzmatrix der Prädiktion, als auch
        die Kovarianzmatrix der Innovation, sowie die neue Designmatrix
        benötigt.
        """
        # Aufstellen der neuen K-Matrix
        self.__k_matrix = self.__kov_x_strich_matrix@np.transpose(self.__a_matrix)@np.linalg.inv(self.__kov_d_matrix)

        # Ausgabe der Berechnungen für eventuelle weitere externe Berechnungen
        return(self.__k_matrix)

    def update(self):
        """
        Beim Update werden die Ergebnisse (X-Dach-Vektor), sowie die
        Genauigkeit der Ergebnisse (Kovarianzmatrix von X-Dach-Vektor)
        aktualisiert.

        Zur Berechnung der neuen Ergebnisse wird die Vorraussage (Prädiktion)
        mit der gewichteten (Gain-Matrix) Innovation (d-Vektor) aktualisiert.
        Außerdem wird der zusätzlichen Hilfe der Designmatrix und der
        Kovarianzmatrix der Prädiktion die Genauigkeit der neuen Ergebnisse
        berechnet (Kovarianzmatrix von X-Dach-Vektor).
        """
        # Aufstellen des neuen X-Dach-Vektors
        self.__x_dach_vektor = self.__x_strich_vektor+self.__k_matrix@self.__d_vektor
        self.__split_x_vektor()

        # Aufstellen der neuen Kovarianzmatrix des X-Dach-Vektors
        self.__kov_x_dach_matrix = self.__k_matrix@self.__a_matrix
        self.__kov_x_dach_matrix = np.identity(len(self.__kov_x_dach_matrix))-self.__kov_x_dach_matrix
        self.__kov_x_dach_matrix = self.__kov_x_dach_matrix@self.__kov_x_strich_matrix

        # Ausgabe der Berechnungen für eventuelle weitere externe Berechnungen
        return(self.__x_dach_vektor, self.__kov_x_dach_matrix)


class DynKalmanFilterGerade():
    """
    Diese Klasse wird zur Anwendung des dynamischen Kalman-Filters benötigt.
    Hierbei wird von einer geradlinigen Bewegungsbahn ausgegangen.
    """
    def __init(self):
        pass

    def set_data_rate(self, data_rate):
        self.__delta_t = data_rate

    def set_start_values(self, x_vektor, kov_x_matrix, kov_w_matrix):
        self.__x_dach_vektor = x_vektor
        self.__kov_x_dach_matrix = kov_x_matrix
        self.__split_x_vektor()
        self.__kov_w_matrix = kov_w_matrix

    def __split_x_vektor(self):
        self.__y = self.__x_dach_vektor[0, 0]
        self.__x = self.__x_dach_vektor[1, 0]
        self.__a = self.__x_dach_vektor[2, 0]
        self.__v = self.__x_dach_vektor[3, 0]

    def praediktion(self):
        """
        Bei der Prädiktion werden die aktuellen Ergebnisse mithilfe der
        Funktionen der Kreisbewegung in die Zukunft prädiziert.
        Die Funktionen werden ohne Linearisierung angewendet!

        Für die neue Transitions- und Störgrößenmatrix werden die Funktionen
        jeweils linearisiert angewendet.

        Außerdem wird die Kovarianzmatrix der Prädiktion berechnet.
        """
        # Aufstellen der Transitionsmatrix
        A = self.__v*m.cos(self.__a)*self.__delta_t
        B = m.sin(self.__a)*self.__delta_t
        C = -self.__v*m.sin(self.__a)*self.__delta_t
        D = m.cos(self.__a)*self.__delta_t
        self.__t_matrix = np.array([[1, 0, A, B],
                                    [0, 1, C, D],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        # Aufstellen der Störgrößenmatrix
        self.__c_matrix = self.__t_matrix[0:, 2:]

        # Prädiktion des nächsten x-Vektors
        y = self.__y+self.__v*m.sin(self.__a)*self.__delta_t
        x = self.__x+self.__v*m.cos(self.__a)*self.__delta_t
        a = self.__a
        v = self.__v
        self.__x_strich_vektor = np.array([[y], [x], [a], [v]])
        del y, x, a, v

        # Berechnung der Kovarianzmatrix der Prädiktion
        self.__kov_x_strich_matrix = self.__t_matrix@self.__kov_x_dach_matrix@np.transpose(self.__t_matrix)+self.__c_matrix@self.__kov_w_matrix@np.transpose(self.__c_matrix)

        # Ausgabe der Berechnungen für eventuelle weitere externe Berechnungen
        return(self.__t_matrix, self.__c_matrix, self.__x_strich_vektor,
               self.__kov_x_strich_matrix)

    def innovation(self, l_vektor_neu, kov_l_matrix_neu):
        """
        Zur Berechnung der Innovation werden die prädizierten Werte und die
        neuen Beobachtungen benötigt, um daraus die Differenz aus Vorhersage
        und Realität (neue Beobachtungen) berechnen zu können.

        Bei der Innovation wird zunächst die neue Designmatrix berechnet und
        dann sowohl die Innovation selbst, als auch die Kovarianzmatrix der
        Innovation berechnet.
        """
        # Aufstellen der Designmatrix
        self.__a_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])

        # Bei Ausfall die Designmatrix und den Beobachtungsvektor verändern
        ausfall_indices = []
        for i, e in enumerate(l_vektor_neu):
            if(np.isnan(e)):
                ausfall_indices.append(i)
        self.__a_matrix = np.delete(self.__a_matrix, ausfall_indices, axis=0)
        l_vektor_neu = np.delete(l_vektor_neu, ausfall_indices, axis=0)
        kov_l_matrix_neu = np.delete(kov_l_matrix_neu, ausfall_indices, axis=1)
        kov_l_matrix_neu = np.delete(kov_l_matrix_neu, ausfall_indices, axis=0)

        # Berechnung der Innovation
        self.__d_vektor = l_vektor_neu-self.__a_matrix@self.__x_strich_vektor

        # Berechnung der Kovarianzmatrix der Innovation
        self.__kov_d_matrix = kov_l_matrix_neu+self.__a_matrix@self.__kov_x_strich_matrix@np.transpose(self.__a_matrix)

        # Ausgabe der Berechnungen für eventuelle weitere externe Berechnungen
        return(self.__a_matrix, self.__d_vektor, self.__kov_d_matrix)

    def gain_matrix(self):
        """
        Die Gain-Matrix stellt die relative Gewichtung der Genauigkeit der
        Vorhersage (Prädiktion) und der jeweiligen Beobachtungen dar.

        Zur Berechnung wird sowohl die Kovarianzmatrix der Prädiktion, als auch
        die Kovarianzmatrix der Innovation, sowie die neue Designmatrix
        benötigt.
        """
        # Aufstellen der neuen K-Matrix
        self.__k_matrix = self.__kov_x_strich_matrix@np.transpose(self.__a_matrix)@np.linalg.inv(self.__kov_d_matrix)

        # Ausgabe der Berechnungen für eventuelle weitere externe Berechnungen
        return(self.__k_matrix)

    def update(self):
        """
        Beim Update werden die Ergebnisse (X-Dach-Vektor), sowie die
        Genauigkeit der Ergebnisse (Kovarianzmatrix von X-Dach-Vektor)
        aktualisiert.

        Zur Berechnung der neuen Ergebnisse wird die Vorraussage (Prädiktion)
        mit der gewichteten (Gain-Matrix) Innovation (d-Vektor) aktualisiert.
        Außerdem wird der zusätzlichen Hilfe der Designmatrix und der
        Kovarianzmatrix der Prädiktion die Genauigkeit der neuen Ergebnisse
        berechnet (Kovarianzmatrix von X-Dach-Vektor).
        """
        # Aufstellen des neuen X-Dach-Vektors
        self.__x_dach_vektor = self.__x_strich_vektor+self.__k_matrix@self.__d_vektor
        self.__split_x_vektor()

        # Aufstellen der neuen Kovarianzmatrix des X-Dach-Vektors
        self.__kov_x_dach_matrix = self.__k_matrix@self.__a_matrix
        self.__kov_x_dach_matrix = np.identity(len(self.__kov_x_dach_matrix))-self.__kov_x_dach_matrix
        self.__kov_x_dach_matrix = self.__kov_x_dach_matrix@self.__kov_x_strich_matrix

        # Ausgabe der Berechnungen für eventuelle weitere externe Berechnungen
        return(self.__x_dach_vektor, self.__kov_x_dach_matrix)

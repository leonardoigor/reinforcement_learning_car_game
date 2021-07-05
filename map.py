import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
import cv2 as cv
# Importação da IA que está no arquivo ai.py
from ai import Dqn

# Não permite adicionar um ponto vermelho no cenário quando é clicado com o botão direito do mouse
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '800')
# As variáveis last_x e last_y são usadas para manter na memória o último ponto quando desenhamos areia no mapa
last_x = 0
last_y = 0
n_points = 0  # Número total de pontos do último desenho
length = 0  # Tamanho do último desenho

# Criamos um objeto que chamamos de brain (cérebro), que contém a rede neural que retorna o valor de Q
# 5 entradas (sensores + direção), 3 saídas e valor de gamma
brain = Dqn(5, 3, 0.9)
# action = 0 => sem rotação, action = 1 => rotaciona 20 graus, action = 2 => rotaciona -20 graus
action2rotation = [0, 20, -20]
last_reward = 0  # inicialização da última recompensa
# inicialização do valor médio das recompensas (sliding window) com relação ao tempo
scores = []

# Inicialização do mapa
first_update = True  # usado para inicializar o mapa somente uma vez


def init():
    global sand  # a areia é representada por um vetor que possui a mesma quantidade de pixels que a interface completo - 1 se tem areia e 0 se não tem areia
    # coordenada x do objetivo (para onde o carro vai, do aeroporto para o centro ou o contrário)
    global goal_x
    # coordenada y do objetivo (para onde o carro vai, do centro para o aeroporto ou o contrário)
    global goal_y
    global first_update
    # inicialização da areia somente com zeros
    sand = np.zeros((longueur, largeur))
    # o objetivo a alcançar é o canto superior esquerdo do mapa (é 20 e não zero porque o carro ganha recompensa negativa se tocar a parede)
    goal_x = 20
    goal_y = largeur - 20  # largura - 20
    first_update = False  # usado para inicializar o mapa somente uma vez


# Inicialização da última distância que indica a distância até o destino
last_distance = 0


class Car(Widget):

    angle = NumericProperty(0)  # inicialização do ângulo do carro
    # inicialização da última rotação do carro (depois de uma ação, o caro faz uma rotação de 0, 20 ou -20 graus)
    rotation = NumericProperty(0)
    # inicialização da coordenada de velocidade x
    velocity_x = NumericProperty(0)
    # inicialização da coordenada de velocidade y
    velocity_y = NumericProperty(0)
    # vetor com a velocidade x e y
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    # inicialização da coordenada x do primeiro sensor (frente)
    sensor1_x = NumericProperty(0)
    # inicialização da coordenadao y do primeiro sensor (frente)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)  # primeiro sensor
    # inicialização da coordenada x do segundo sensor (30 graus para a esquerda)
    sensor2_x = NumericProperty(0)
    # inicialização da coordenada y do segundo sensor (30 graus para a esquerda)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)  # segundo sensor
    # inicialização da coordenada x do terceiro sensor (30 graus para a direita)
    sensor3_x = NumericProperty(0)
    # inicialização da coordenada y do terceiro sensor (30 graus para a direita)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)  # terceiro sensor
    # inicialização do sinal recebido pelo sensor 1
    signal1 = NumericProperty(0)
    # inicialização do sinal recebido pelo sensor 2
    signal2 = NumericProperty(0)
    # inicialização do sinal recebido pelo sensor 3
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # atualiza a posição do carro de acordo com sua última posição e velocidade
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation  # busca a rotação do carro
        self.angle = self.angle + self.rotation  # atualiza o ângulo
        self.sensor1 = Vector(30, 0).rotate(self.angle) + \
            self.pos  # atualiza a posição do sensor 1
        self.sensor2 = Vector(30, 0).rotate((self.angle+30) %
                                            360) + self.pos  # atualiza a posição do sensor 2
        self.sensor3 = Vector(30, 0).rotate((self.angle-30) %
                                            360) + self.pos  # atualiza a posição do sensor 3
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(
            self.sensor1_y)+10]))/400.  # calcula o sinal recebido do sensor 1 (densidade de areia ao redor do sensor 1)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(
            self.sensor2_y)+10]))/400.  # calcula o sinal recebido do sensor 1 (densidade de areia ao redor do sensor 2)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(
            self.sensor3_y)+10]))/400.  # calcula o sinal recebido do sensor 1 (densidade de areia ao redor do sensor 3)
        if self.sensor1_x > longueur-10 or self.sensor1_x < 10 or self.sensor1_y > largeur-10 or self.sensor1_y < 10:  # se o sensor 1 saiu do mapa
            self.signal1 = 1.  # sensor 1 detecta areia
        if self.sensor2_x > longueur-10 or self.sensor2_x < 10 or self.sensor2_y > largeur-10 or self.sensor2_y < 10:  # se o sensor 2 saiu do mapa
            self.signal2 = 1.  # sensor 2 detecta areia
        if self.sensor3_x > longueur-10 or self.sensor3_x < 10 or self.sensor3_y > largeur-10 or self.sensor3_y < 10:  # se o sensor 3 saiu do mapa
            self.signal3 = 1.  # sensor 3 detecta areia


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


class Game(Widget):

    car = ObjectProperty(None)  # busca o objeto carro do arquivo kivy
    ball1 = ObjectProperty(None)  # busca o objeto sensor 1 do arquivo kivy
    ball2 = ObjectProperty(None)  # busca o objeto sensor 2 do arquivo kivy
    ball3 = ObjectProperty(None)  # busca o objeto sensor 3 do arquivo kivy

    def serve_car(self):  # inicia o carro quando executamos a aplicação
        self.car.center = self.center  # posição inicial do carro no centro do maps
        # o carro começa se movendo na horizontal e com velocidade 6
        self.car.velocity = Vector(6, 0)

    def update(self, dt):  # função que atualiza todas as variáveis a cada tempo t, quando chega em um novo estado (pega novos valores dos sensores)
        # especificações das variáveis globais
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width  # largura do mapa (horizontal)
        largeur = self.height  # altura do mapa (vertical)
        if first_update:  # inicialização do mapa somente uma vez
            init()

        # diferença da coordenada x entre o objetivo e onde o carro está agora
        xx = goal_x - self.car.x
        # diferença da coordenada y entre o objetivo e onde o carro está agora
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / \
            180.  # direção do carro com relação ao objetivo (se o carro está apontando perfeitamente para o objetivo a orientação é igual a zero
        # esse é o vetor de entrada, composto por três sinais dos sensores mais a orientação positiva e negativa
        last_signal = [self.car.signal1, self.car.signal2,
                       self.car.signal3, orientation, -orientation]
        # a rede neural vai indicar a próxima ação
        action = brain.update(last_reward, last_signal)
        # adiciona os valores das recompensas (média das 1000 últimas recompensas - sliding window)
        scores.append(brain.score())
        # converte a ação atual (0, 1 or 2) nos ângulos de rotação (0°, 20° ou -20°)
        rotation = action2rotation[action]
        self.car.move(rotation)  # move o carro baseado na rotação
        # calcula a nova distância entre o carro e o objetivo
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1  # atualiza a posição do sensor 1
        self.ball2.pos = self.car.sensor2  # atualiza a posição do sensor 2
        self.ball3.pos = self.car.sensor3  # atualiza a posição do sensor 3

        # plt.show()

        if sand[int(self.car.x), int(self.car.y)] > 0:  # se o carro está na areia
            self.car.velocity = Vector(1, 0).rotate(
                self.car.angle)  # diminui a velocidade de 6 para 1
            last_reward = -1  # ganha uma recompensa negativa
        else:  # caso contrário, se não estiver na areia
            self.car.velocity = Vector(6, 0).rotate(
                self.car.angle)  # mantém a velocidade padrão de 6
            last_reward = -0.2  # ganha uma recompensa negativa pequena
            if distance < last_distance:  # caso esteja chegando no objetivo
                last_reward = 0.1  # ganha uma pequena recompensa positiva

        if self.car.x < 10:  # se o carro está no canto esquerdo
            self.car.x = 10  # posiciona o carro perto da parede
            last_reward = -1  # ganha recompensa negativa
        if self.car.x > self.width - 10:  # se o carro está no canto direito
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:  # se o carro está na borda inferior
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:  # se o carro está na borda superior
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:  # quando o carro chega no objetivo
            # o objetivo muda para o canto inferior direito (e vice-versa), atualizando x e y
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y

        # Atualiza a distância para o objetivo
        last_distance = distance

# Interface gráfica (veja https://kivy.org/docs/tutorials/firstwidget.html)


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):  # adiciona areia quando clicamos com o botão esquerdo
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    # adiciona areia quando movemos o mouse enquanto pressionamos
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10,
                 int(touch.y) - 10: int(touch.y) + 10] = 1
            last_x = x
            last_y = y


class CarApp(App):

    def build(self):  # building the app
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):  # clear button
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):  # save button
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):  # load button
        print("loading last saved brain...")
        brain.load()


# Execução de todo o código
if __name__ == '__main__':
    CarApp().run()

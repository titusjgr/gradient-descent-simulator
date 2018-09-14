# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pygame
import types

# 從網路上載入使用的資料
def default_datapoints_generate(number_of_points):
#    '''回傳一個ndarray，[0]是rooms per person，[1]是median house value'''
    read_in_data = pd.read_csv("california_housing_train.csv", sep=",")
    #   print(read_in_data)
    ##   print(read_in_data)
    #read_in_data = read_in_data.head(5)
    read_in_data = np.asarray(((read_in_data['total_rooms'] / read_in_data['population']).apply(lambda x: min(x, 5)),
                              read_in_data['median_house_value']))

#    print('original data')
#    print(read_in_data[0], read_in_data[1])
#    print(np.max(read_in_data[0]), np.min(read_in_data[0]))
#    print(np.max(read_in_data[1]), np.min(read_in_data[1]))
#    print('standardized')
    read_in_data[0] = (read_in_data[0] - np.mean(read_in_data[0])) / np.std(read_in_data[0])
    read_in_data[1] = (read_in_data[1] - np.mean(read_in_data[1])) / np.std(read_in_data[1])
#    print(np.max(read_in_data[0]), np.min(read_in_data[0]))
#    print(np.max(read_in_data[1]), np.min(read_in_data[1]))
#    print('fitting')
    read_in_data[0] = (read_in_data[0] - np.min(read_in_data[0])) / np.max(read_in_data[0]) * 230
    read_in_data[1] = (read_in_data[1] - np.min(read_in_data[1])) / np.max(read_in_data[1]) * 240
#    print(np.max(read_in_data[0]), np.min(read_in_data[0]))
#    print(np.max(read_in_data[1]), np.min(read_in_data[1]))
    print('choose {} datapoints'.format(number_of_points))

    return_data = np.random.choice(17001, number_of_points, False)
    return_data = tuple((tuple((read_in_data[0, num] for num in return_data)),
                         tuple((read_in_data[1, num] for num in return_data))))
    return_data = np.asarray(return_data)
#    print(return_data)
#    print(return_data)
#    plt.scatter(return_data[:,0], return_data[:,1])


    return return_data
##################
#初始字幕高度
th = -40
##################
# 顏色
black = (0,0,0)
white = (255,255,255)
red = (200,0,0)
green = (0,200,0)
bright_red = (255,0,0)
bright_green = (0,255,0)

# 字體
ttf = 'freesansbold.ttf'


class Game:
    def __init__(self, display_width=800, display_height=600, fps=60):
        # 螢幕初始化
        pygame.init()
        pygame.display.set_caption('Gradient Descent Game')
        self.display_width = display_width
        self.display_height= display_height
        self.screen = pygame.display.set_mode((display_width, display_height))

        self.clock = pygame.time.Clock()
        self.fps = fps
        self.running = True # 迴圈
        self.continuing = True # 進入下個迴圈
        self.pressed = None # 被按下的按鍵

        self.chart_sidelength = 400 # 製作資料圖時使用

        self.cost_list = [] # 最後畫圖






#                                         字體中心點預設是在視窗中心
#                             文字  大小  水平偏移   垂直偏移
    def message_display(self, text, size, hshift=0, vshift=0, color=black):
        text_surf, text_rect = self.text_objects(text, pygame.font.Font(ttf, size), color)
        text_rect.center = ((self.display_width / 2 + hshift), (self.display_height / 2 + vshift))
        self.screen.blit(text_surf, text_rect)


    @staticmethod
    def text_objects(text, font, color):
        text_surf = font.render(text, True, color)
        return text_surf, text_surf.get_rect()

    def draw_points(self, color, x, y): # 轉換座標系統，以及畫出資料點，從數學的x,y轉換成視覺上符合的樣子
        x = int(np.ma.round(self.lbound + x))
        y = int(np.ma.round(self.bbound - y))

        pygame.draw.circle(self.screen, color, (x, y), 3)



#   開始畫面
    def start_menu(self):
        self.loop_dict = {pygame.K_c: self.choose_data, pygame.K_g: self.generate_data, pygame.K_ESCAPE: None}
        text_size_title = 80
        text_size_small = 24
        next_loop = None


        while self.running == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.continuing = False
                if event.type == pygame.KEYDOWN:
                    self.pressed = event.key
                    if self.pressed in self.loop_dict:
                        self.running = False
                        self.continuing = True
                        next_loop = self.loop_dict[self.pressed]
#                        print(self.pressed)


            self.screen.fill(white)

#           遊戲名稱
            self.message_display('Gradient', text_size_title, vshift=-text_size_title)
            self.message_display('Descent', text_size_title)
            self.message_display('Game', text_size_title, vshift=text_size_title)

            self.message_display('"C"hoose the points yourself or let the program "G"enerate for you.',
                                 text_size_small, vshift = self.display_height / 2 - text_size_small * 2)

            pygame.display.update()
            self.clock.tick(self.fps)

        self.screen.fill(white)
        self.lbound = (self.display_width - self.chart_sidelength) / 2
        self.rbound = (self.display_width + self.chart_sidelength) / 2
        self.tbound = (self.display_height - self.chart_sidelength) / 2
        self.bbound = (self.display_height + self.chart_sidelength) / 2
        pygame.draw.rect(self.screen, black, (self.lbound, self.tbound,
                                               self.chart_sidelength, self.chart_sidelength), 1)
        if self.continuing == True:
            return next_loop


#       自行選擇資料點
    def choose_data(self):
        text_size_small = 24
        text_displayed = False
        self.loop_dict = {pygame.K_ESCAPE: None,
                          pygame.K_RETURN: self.get_learning_rate, pygame.K_KP_ENTER: self.get_learning_rate}
        pygame.display.update()
        self.running = True
        self.clicked = False
        self.datapoints = []

        while self.running == True:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.continuing = False
                if event.type == pygame.KEYDOWN:
                    self.pressed = event.key
                    if self.pressed in self.loop_dict:
                        self.running = False
                        self.continuing = True
                        next_loop = self.loop_dict[self.pressed]
#                        print(self.pressed)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.clicked == False: # 目前沒有觸發
                        if self.lbound < mouse_pos[0] < self.rbound and self.tbound < mouse_pos[1] < self.bbound:
#                            print(mouse_pos)
                            self.datapoints.append((mouse_pos[0] - self.lbound, -(mouse_pos[1] - self.bbound)))

                            self.draw_points(black, *self.datapoints[-1])
                            if text_displayed == False:
                                self.message_display('Press ENTER to continue.', text_size_small,
                                                     vshift = self.display_height / 2 - text_size_small * 2)
                                text_displayed = True

                    self.clicked = True
                else:
                    self.clicked = False

            self.clock.tick(self.fps)
            pygame.display.update()


        if self.continuing == True:
            self.datapoints = np.asarray(tuple(tuple(point[i] for point in self.datapoints) for i in range(2)))
#            print(self.datapoints)

            return next_loop


    def draw_all_points(self):
        for index in range(self.datapoints.shape[1]):
            self.draw_points(black, self.datapoints[0, index], self.datapoints[1, index])

#       自動產生資料點
    def generate_data(self):
        text_size_small = 24
        self.loop_dict = {pygame.K_ESCAPE: None,
                          pygame.K_RETURN: self.get_learning_rate, pygame.K_KP_ENTER: self.get_learning_rate}
        pygame.display.update()
        self.running = True

        self.datapoints = default_datapoints_generate(10)
        self.draw_all_points()
        self.message_display('Press ENTER to continue.', text_size_small,
                             vshift = self.display_height / 2 - text_size_small * 2)

        while self.running == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.continuing = False
                if event.type == pygame.KEYDOWN:
                    self.pressed = event.key
                    if self.pressed in self.loop_dict:
                        self.running = False
                        self.continuing = True
                        next_loop = self.loop_dict[self.pressed]
#                        print(self.pressed)

            self.clock.tick(self.fps)
            pygame.display.update()


        if self.continuing == True:
            return next_loop

#       輸入 Learning Rate
    def get_learning_rate(self):
        text_size_small = 24
        self.loop_dict = {pygame.K_ESCAPE: None,
                          pygame.K_RETURN: self.set_stepspeed, pygame.K_KP_ENTER: self.set_stepspeed} ## 再增加
        self.running = True

        self.input = ""

        while self.running == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.continuing = False
                if event.type == pygame.KEYDOWN:
                    self.pressed = event.key
                    if self.pressed in self.loop_dict:
                        self.running = False
                        self.continuing = True
                        next_loop = self.loop_dict[self.pressed]

                     # 對應 0 ~ 9 和 .
                    if 48 <= self.pressed <= 57: # 主區的 0 到 9
                        self.input += str(self.pressed - 48)
                    elif 256 <= self.pressed <= 265: # 數字區的 0 到 9
                        self.input += str(self.pressed - 256)
                    elif self.pressed == 46 or self.pressed == 266: # .
                        self.input += '.'
                    elif self.pressed == 8: # backspace
                        self.input = self.input[0: -1]


            pygame.draw.rect(self.screen, white, (0, self.bbound + 3, self.display_width, 100))
            self.message_display('Learning rate:', text_size_small,
                                 vshift = self.display_height / 2 - text_size_small * 3)
            self.message_display(self.input, text_size_small,
                                 vshift = self.display_height / 2 - text_size_small * 2)
            self.message_display('Press ENTER to continue.', text_size_small,
                                 vshift = self.display_height / 2 - text_size_small * 1)

            self.clock.tick(self.fps)
            pygame.display.update()


        try:
            self.learning_rate = float(self.input)
#            print(self.learning_rate)
        except ValueError:
            return self.get_learning_rate

        if self.continuing == True:
            return next_loop



    def set_stepspeed(self):
        text_size_small = 24
        self.loop_dict = {pygame.K_ESCAPE: None,
                          pygame.K_RETURN: self.gd_step1, pygame.K_KP_ENTER: self.gd_step1}
        self.running = True
        self.input = ""

        while self.running == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.continuing = False
                if event.type == pygame.KEYDOWN:
                    self.pressed = event.key
                    if self.pressed in self.loop_dict:
                        self.running = False
                        self.continuing = True
                        next_loop = self.loop_dict[self.pressed]


                    # 對應 0 ~ 9 和 .
                    elif 48 <= self.pressed <= 57: # 主區的 0 到 9
                        self.input += str(self.pressed - 48)
                    elif 256 <= self.pressed <= 265: # 數字區的 0 到 9
                        self.input += str(self.pressed - 256)
                    elif self.pressed == 46 or self.pressed == 266: # .
                        self.input += '.'
                    elif self.pressed == 8: # backspace
                        self.input = self.input[0: -1]


            pygame.draw.rect(self.screen, white, (0, self.bbound + 3, self.display_width, 100))
            self.message_display('Time per step: (s)', text_size_small,
                                 vshift = self.display_height / 2 - text_size_small * 3)
            self.message_display(self.input, text_size_small,
                                 vshift = self.display_height / 2 - text_size_small * 2)
            self.message_display('Press ENTER to continue.', text_size_small,
                                 vshift = self.display_height / 2 - text_size_small * 1)

            self.clock.tick(self.fps)
            pygame.display.update()


        try:
            self.wait_ms = float(self.input)
            self.wait_ms *= 1000
#            print(self.wait_ms)
        except ValueError:
            if self.continuing != False:
                return self.set_stepspeed

        if self.continuing == True:
            self.w = np.zeros((2)) # 已經是 1 * 2 矩陣
            self.x = np.asarray((np.ones(self.datapoints.shape[1]), self.datapoints[0]))
            self.x[1] = (self.x[1] - np.mean(self.x[1])) / np.std(self.x[1]) # 標準化

            # 將標準化的資料轉為繪圖適當大小#############3
            self.shift = 0 - np.min(self.x[1])
            self.datapoints[0] = self.x[1] + self.shift
            self.scale = 400 / np.max(self.datapoints[0]) * 0.98 # 避免誤差造成出框
            self.datapoints[0] *= self.scale


            self.y = np.asarray(self.datapoints[1])
#            print(self.w, self.x, self.y, end='\n')

            self.mse = self.calculate_cost(self.w, self.x, self.y)
#            print(self.mse)


            pygame.time.wait(2000)

            return next_loop


    def draw_line(self):
        line_list = [] # 應該含有兩個點
        a = self.a
        b = self.b

        if a == 0: #避免 ZeroDivisionError
            line_list = [[0, b], [400, b]]
        else:
            if 400 >= b >= 0:
                line_list.append([0, b])
            if 400 >= 400 * a + b >= 0:
                line_list.append([400, 400 * a + b])
            if 400 > -b / a > 0:
                line_list.append([-b / a, 0])
            if 400 > (400 - b) / a > 0:
                line_list.append([(400 - b) / a, 400])

        if len(line_list) < 2:
            return None
        line_list[0][0] = int(np.ma.round(self.lbound + line_list[0][0]))
        line_list[1][0] = int(np.ma.round(self.lbound + line_list[1][0]))
        line_list[0][1] = int(np.ma.round(self.bbound - line_list[0][1]))
        line_list[1][1] = int(np.ma.round(self.bbound - line_list[1][1]))

        pygame.draw.aaline(self.screen, black, line_list[0], line_list[1], 3)



    def show_distance(self, pair):
        pygame.draw.line(self.screen, black, (self.lbound + pair[0], self.bbound - pair[1]),
                         (self.lbound + pair[0], self.bbound - self.predict(pair[0])), 1)

    def predict(self, input_x):
        return self.a * input_x + self.b

    def calculate_cost(self, w, x, y):
        return np.sum(np.power(np.matmul(w, x) - y, 2)) / len(y)

    def draw_chart(self):
        pygame.draw.rect(self.screen, black, (self.lbound, self.tbound,
                                               self.chart_sidelength, self.chart_sidelength), 1)
        for index in range(self.datapoints.shape[1]):
            self.draw_points(black, self.datapoints[0, index], self.datapoints[1, index])
        self.draw_line()

        # 展示距離
    def gd_step1(self):
        self.running = True
        self.loop_dict = {
                pygame.K_ESCAPE: None,
                pygame.K_RETURN: self.gd_end,
                pygame.K_RETURN: self.gd_end
                }

        times = 0

        self.a = self.w[1] / self.scale
        self.b = self.w[0] - self.a * self.shift * self.scale

        while self.running == True and times < len(self.y):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.continuing = False
                if event.type == pygame.KEYDOWN:
                    self.pressed = event.key

            for item in self.loop_dict.items():
                if pygame.key.get_pressed()[item[0]] == True:
                    next_loop = item[1]
                    self.running = False

            self.screen.fill(white)
            self.draw_chart()

            self.draw_line()
            self.draw_points(green, self.datapoints[0, times], self.datapoints[1, times])

            self.draw_points(red, self.datapoints[0, times], self.predict(self.datapoints[0, times]))
            self.show_distance(self.datapoints[:, times])

            pygame.display.update()
            pygame.time.wait(int(self.wait_ms / len(self.y)))
            times += 1

        next_loop = self.gd_step2
        if self.continuing == True:
            return next_loop

        # 顯示 Cost
    def gd_step2(self):
        text_size_small = 24
        self.running = True
        next_loop = self.gd_step3
        self.loop_dict = {
                pygame.K_ESCAPE: None,
                pygame.K_RETURN: self.gd_end,
                pygame.K_RETURN: self.gd_end
                }

        self.mse = self.calculate_cost(self.w, self.x, self.y)
        self.cost_list.append(self.mse)

        self.screen.fill(white)
        self.draw_chart()
        self.draw_all_points()
        self.message_display('Currenct cost:', text_size_small,
                             vshift = self.display_height / 2 - text_size_small * 3)
        self.message_display(str(self.mse), text_size_small,
                             vshift = self.display_height / 2 - text_size_small * 2)
        self.message_display('Press ENTER to continue.', text_size_small,
                             vshift = self.display_height / 2 - text_size_small * 1)

        pygame.display.update()
        for item in self.loop_dict.items():
                if pygame.key.get_pressed()[item[0]] == True:
                    next_loop = item[1]

        if self.continuing == True:
            return next_loop


        # 重畫直線

    @staticmethod
    def plot_config():
        plt.title('Cost During Iteration')
        plt.xlabel('Num of Iterations')
        plt.ylabel('Cost')
        plt.ylim(ymin=0)
        plt.show()


    def gd_step3(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.continuing = False
            if event.type == pygame.KEYDOWN:
                self.pressed = event.key
                if self.pressed == pygame.K_ESCAPE:
                    self.running = False
                    self.continuing = False
                if self.pressed == pygame.K_RETURN or self.pressed == pygame.K_KP_ENTER:
                    self.running = False
                    self.continuing = True
                    return self.gd_end

        update = self.learning_rate * np.matmul(np.matmul(self.w, self.x) - self.y, np.transpose(self.x))
        self.w -= update
        # print(update, self.w, 'jizz')


        plt.plot(range(max(0, len(self.cost_list) - 5), len(self.cost_list)), self.cost_list[-1:-6:-1][::-1], '.-')
        # self.plot_config()
        pygame.time.wait(int(self.wait_ms))
        plt.close()

        return self.gd_step1


    def gd_end(self):
#        print(self.cost_list)
        plt.plot(self.cost_list, '.-')
        self.plot_config()

##################

game = Game()
now_executing = game.start_menu()



while isinstance(now_executing, types.MethodType):
    now_executing = now_executing()
pygame.quit()

##################
# 測試
#default_datapoints_generate(7)

class Tamagochi:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.hunger = 100
        self.level = 1  # Добавляем уровень
        self.defense = 10  # Процент защиты octupusa

    def hit(self, damage): # задаём урон с учётом защиты
            # Уменьшаем урон на процент защиты
            damage_small = damage * (1 - self.defense / 100)
            self.health -= damage_small

    # Задаём урон
    def hit(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            print("Умер. Игра окончена.")
            exit()
        self.hunger -= 5
        if self.hunger > 100:
            self.hunger = 100
        if self.hunger < 0:
            self.hunger = 0
            print('Умер от голода. Игра окончена')
            exit()

    # Задаём хп
    def heal(self, hp):
        self.health += hp
        if self.health > 100:
            self.health = 100
        self.hunger += 5
        if self.hunger > 100:
            self.hunger = 100

    # Задаём кол-во еды
    def feed(self, food):
        self.hunger += food  # Увеличиваем сытость
        if self.hunger > 100:
            self.hunger = 100
        self.health += 5
        if self.health > 100:
            self.health = 100

    def level_up(self):
        if self.level < 3:  # Максимальный уровень 3
            self.level += 1  # Увеличиваем уровень
            self.health += 75  # Увеличиваем живучесть
            self.hunger -= 10  # Уменьшаем голод
            self.defense += 5  # Увеличиваем защиту на 5% при повышении уровня
            if self.hunger < 0:
                self.hunger = 0
            self.update_view()  # Обновляем внешний вид
            print(f"{self.name} достиг уровня {self.level}!")
        else:
            print(f"{self.name} уже максимального уровня!")

    def __str__(self):
        return (f"{self.view}\n{self.name}: Уровень={self.level}, Здоровье={self.health}, Голод={self.hunger}, Защита={self.defense}%"
        f"")



class Octopus(Tamagochi):
    def __init__(self, name):
        super().__init__(name)
        self.update_view()

    def update_view(self):
        if self.level == 1:
            self.view = r"""
              ,---.
             ( @ @ )
              ).-.(
             '/|||\`
               '|`   
            """
        elif self.level == 2:
            self.view = r"""
              ,-----.
             ( @   @ )
              )  ---  (
             '/|||||||\`
               '|||`   
            """
        elif self.level == 3:
            self.view = r"""
              ,-------.
             ( @     @ )
              )  ---  (
             '/|||||||\`
               '|||||`   
            """


class Cat(Tamagochi):
    def __init__(self, name):
        super().__init__(name)
        self.update_view()

    def update_view(self):
        if self.level == 1:
            self.view = r"""
              /\_/\  
             ( o.o ) 
              > ^ <
            """
        elif self.level == 2:
            self.view = r"""
("`-''-/").___..--''"`-._ 
 `6_ 6 ) `-. ( ).`-.__.`) 
 (_Y_.)' ._ ) `._ `. ``-..-' 
 _..`--'_..-_/ /--'_.'
 ((((.-'' ((((.' (((.-' 
            """
    def level_up(self):
        if self.level < 2:
            self.level += 1  # Увеличиваем уровень
            self.health += 125  # Увеличиваем живучесть
            self.defense += 10  # Увеличиваем защиту на 10% при повышении уровня
            self.hunger -= 25  # Уменьшаем голод
            if self.hunger < 0:
                self.hunger = 0
            self.update_view()  # Обновляем внешний вид
            print(f"{self.name} достиг уровня {self.level}!")
        else:
            print(f"{self.name} уже максимального уровня!")


if __name__ == "__main__":
    name = input("Введите имя осьминога: ")
    octo = Octopus(name)
    name_cat = input("Введите имя кота: ")
    cat = Cat(name_cat)

    cotik_draw = octo  # По умолчанию выбран осьминог
    print(cotik_draw)

    while True:
        action = input("Выберите действие:\n q - выход, b - бить, h - лечить, f - кормить, l - повысить уровень, ""s - переключиться на осьминога, c - переключиться на кота\n")
        if action == 'q':
            break
        elif action == 'b':
            damage = int(input("Введите урон: "))
            cotik_draw.hit(damage)
            print(cotik_draw)
        elif action == "h":
            hp = int(input("Введите хп: "))
            cotik_draw.heal(hp)
            print(cotik_draw)
        elif action == "f":
            food = int(input("Введите сытость: "))
            cotik_draw.feed(food)
            print(cotik_draw)
        elif action == "l":
            cotik_draw.level_up()
            print(cotik_draw)
        elif action == "s":
            cotik_draw = octo
            print(f"Переключились на осьминога {octo.name}.")
            print(cotik_draw)
        elif action == "c":
            cotik_draw = cat
            print(f"Переключились на кота {cat.name}.")
            print(cotik_draw)
        else:
            print("Неверный ввод")
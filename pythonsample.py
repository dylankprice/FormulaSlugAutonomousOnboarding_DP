import random
game = True
while game is True:
    dice_num1 = random.randint(1, 6)
    dice_num2 = random.randint(1, 6)
    answer = input("Roll the dice? (y/n): ")
    if answer == "y":
        print(f"({dice_num1}, {dice_num2})")
    elif answer == "n":
        print("Thanks for playing!")
        game = False
    else:
        print("Invalid choice!")
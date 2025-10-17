result = 0
user_in = int(input())

while user_in >= 0:
    if user_in % 5 == 0:
        print("win")
        result = result + 1
    else:
        print('lose')
    user_in = int(input())
    
print(f"Result is {result}")
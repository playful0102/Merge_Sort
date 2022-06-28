itemsArray = []
itemsNumArray = []
itemsPriceArray = []
while True:
    item = input("items want to sell")
    if item == "no":
        break
    else:
        itemsArray.append(item)
        itemNum = int(input("how many items"))
        itemsNumArray.append(itemNum)
        itemPrice = int(input("price in c"))
        itemsPriceArray.append(itemPrice)

print("WTS Softcore")
for i in range(len(itemsArray)):
    print(itemsArray[i] + " " + str(itemsNumArray[i]) + "  " + str(itemsPriceArray[i])+"c")
print("IGN: playfulHogRider")

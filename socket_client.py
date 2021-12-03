# -*- coding: UTF-8 -*-

import socket



client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client.connect(('localhost', 6688))


welcome_message = client.recv(1024).decode('utf-8')
print(welcome_message)
greeting = client.recv(1024).decode('utf-8')
print(greeting)


while True:
    print("you:",end='')
    data = input()
    
    if data.lower() == 'quit':
        break
    data = data.encode('utf-8')
  
    client.sendall(data)
    
    rec_data = client.recv(1024)
    rec_data = rec_data.decode('utf-8')
    print("Chat bot: "+rec_data)

client.close()
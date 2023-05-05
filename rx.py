from unetpy import UnetSocket
print("--rxc begin--")
s2 = UnetSocket('localhost', 1102)                               
rx = s2.receive()                                                
print('from node', rx.from_, ':', bytearray(rx.data).decode())  
s2.close()
print("--rxc stop--")